"""账本接线：判断一拍把承诺写盘（复用的一拍不写），结算只认归约的定稿日，账本损坏不拖着关联。"""

from __future__ import annotations

import asyncio
from datetime import UTC, timedelta

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.judge import CandidateVerdict, Judgement
from habitus.foresight.ledger import verified_counts
from habitus.foundation.observability import ObservationEvent
from habitus.runtime.foresight_ledger import ForesightLedgerStore
from habitus.runtime.foresight_settlement import SettlementStage
from habitus.scene import FactKey
from tests.unit.foresight.fixtures import ScriptedJudge
from tests.unit.runtime.test_foresight_wiring import EVENING, assembled, scripted_judge

#: 现场里每个周一 19:00 打球；说话要在那之前，19:00 才落在承诺的时窗（第 76 槽）里。
SPOKEN = EVENING.replace(hour=18, minute=50)


class Recorder:
    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def promising_judge(pack) -> ScriptedJudge:
    """对着真实的包造一个判「打球 会 76–77」的判断者：引用第一张卡。"""

    play = next(item for item in pack.expanded if item.kind_token == "打球")
    verdict = CandidateVerdict(
        kind_token="打球", verdict="会", window=(76, 77), next=(), basis=(play.background.cards[0].uri,), note="像"
    )
    script = Judgement(
        judged_at=SPOKEN.astimezone(UTC),
        generation="script",
        moment=pack.moment,
        verdicts=(verdict,),
        day_state="正常",
        day_note=None,
        judge_version="scripted-judge",
    )
    return ScriptedJudge(script)


def wired(tmp_path, *, closed_days=tuple, facts=None):
    """真实组装一份预测层，判断者对着真实的包判「会」；结算的定稿日与事实门由调用方给。"""

    _config, probe = assembled(tmp_path / "probe", judge=scripted_judge())
    pack = probe.assembler.assemble(now=SPOKEN)
    config, components = assembled(
        tmp_path / "real", judge=promising_judge(pack), closed_days=closed_days, facts=facts
    )
    return config, components


def test_a_new_judgement_writes_its_promises_and_a_reused_one_does_not(tmp_path) -> None:
    config, components = wired(tmp_path)
    ledger = components.runner.ledger
    assert isinstance(ledger, ForesightLedgerStore) and ledger.root == config.foresight_root

    first = asyncio.run(components.runner.run_once(now=SPOKEN))
    assert not first.reused and [c.kind_token for c in first.claims] == ["打球"]
    # 还没有任何真实的条件源：默认是显式的空实现，承诺照记、三样都是空的。
    assert (first.claims[0].conditions, first.claims[0].condition_keys, first.claims[0].facts_version) == ((), (), "none")
    assert [c.claim_id for c in ledger.claims_on(SPOKEN.date())] == [first.claims[0].claim_id]
    again = asyncio.run(components.runner.run_once(now=SPOKEN + timedelta(minutes=2)))
    assert again.reused and again.claims == () and len(ledger.claims_on(SPOKEN.date())) == 1
    # 同一份承诺重写幂等；内容不同才是损坏。
    assert ledger.record_claim(first.claims[0]) is False


def test_settlement_waits_for_the_day_the_reduction_calls_closed(tmp_path) -> None:
    day = SPOKEN.date()
    closed: list = []
    _config, components = wired(tmp_path, closed_days=lambda: tuple(closed))
    run = asyncio.run(components.runner.run_once(now=SPOKEN))
    observer = Recorder()
    components.settlement.observer = observer  # type: ignore[assignment]

    # 归约还没说今天定稿：留着，不结。树上已经有今天的行也不算——定稿只有归约说了算。
    report = components.settlement.run_once()
    assert (report.settled, report.pending_days) == (0, (day,))
    assert components.runner.ledger.unsettled_claims(day) == run.claims

    closed.append(day)
    report = components.settlement.run_once()
    assert (report.settled, report.verified, report.pending_days) == (1, 1, ())
    (item,) = components.runner.ledger.settlements_on(day)
    assert item.verified and item.claim_id == run.claims[0].claim_id
    assert verified_counts(components.runner.ledger.settlements()) == {("打球", ""): 1}
    # 再结一次什么都不做；两拍都记了观测。
    assert components.settlement.run_once().settled == 0
    assert [(e.category, e.operation, e.status.value) for e in observer.events] == [("foresight", "settlement", "success")] * 3
    assert observer.events[1].attributes["verified"] == 1


def test_a_broken_ledger_is_observed_and_does_not_stop_the_association_stage(tmp_path) -> None:
    """结算失败自己留观测；夜批把它吞掉，关联照常跑——一个读不出来的文件不该让语义层整夜停摆。"""

    from habitus.runtime.assembly import _nightly_stages

    class Broken(SettlementStage):
        def __init__(self) -> None:
            self.observer = Recorder()  # type: ignore[assignment]

        def run_once(self):  # type: ignore[override]
            raise ForesightError("ledger record cannot be read")

    ran: list[str] = []

    async def association() -> str:
        ran.append("association")
        return "ok"

    hook = _nightly_stages(Broken(), association)
    assert hook is not None
    assert asyncio.run(hook()) == "ok" and ran == ["association"]
    # 关联自己的失败仍然往上抛（worker 记账），不被吞。
    async def failing() -> object:
        raise RuntimeError("association broke")

    with pytest.raises(RuntimeError, match="association broke"):
        asyncio.run(_nightly_stages(None, failing)())  # type: ignore[misc]


class Weather:
    """一个只答一个键的源，用来验条件确实经承诺落盘。"""

    version = "weather_v1"

    def keys(self):
        return (FactKey("天气.天象", "类别"),)

    def at(self, when):
        return (("天气.天象", "晴"),)


class Down(Weather):
    def at(self, when):
        raise RuntimeError("weather source down")


def test_an_injected_source_reaches_the_claim_and_a_broken_one_does_not_cost_the_tick(tmp_path) -> None:
    """条件是记在旁边的旁证：源坏了记成"没问到、口径 unavailable"，模型那一拍的判断与承诺照样留下。"""

    _config, components = wired(tmp_path / "ok", facts=Weather())
    run = asyncio.run(components.runner.run_once(now=SPOKEN))
    (claim,) = run.claims
    assert (claim.conditions, claim.condition_keys, claim.facts_version) == ((("天气.天象", "晴"),), ("天气.天象",), "weather_v1")

    _config, broken = wired(tmp_path / "down", facts=Down())
    run = asyncio.run(broken.runner.run_once(now=SPOKEN))
    (claim,) = run.claims
    assert (claim.conditions, claim.condition_keys, claim.facts_version) == ((), (), "unavailable")
    assert broken.runner.last is not None and len(broken.runner.ledger.claims_on(SPOKEN.date())) == 1
