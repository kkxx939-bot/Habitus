"""一条 lane 的先验：两份缓存的读写与归题（实时化阶段 2「先验混合问法」）。

- 旧问法的答案按后果存（``advice/priors``，格式不变，盘上已有的照用）；新问法的按前因另存一份
  （``advice/priors-by-antecedent``）。取一对的档：新问法那份里有就用它，否则用旧问法那份。
- 归题的规则在领域包（``scene.advice.pairing``）：只看两个概念出现的先后。这里取材料——概念出现的那一天（创建时刻的当地
  日期；``settled`` 之外的是还没进概念集的，算最晚、同一批）、哪些对子已有答案——调用、落盘。
- 缓存键里一个概念的那一份是给模型看的名字与判据：改名或改判据就是另一个概念，涉及它的对子重新归题、重问。
"""

from __future__ import annotations

from collections.abc import Container
from datetime import UTC, date, tzinfo

from habitus.foundation.integrity import canonical_digest
from habitus.scene.advice.model import PriorLevel
from habitus.scene.advice.pairing import Questions, assign
from habitus.scene.advice.prior import PriorAdvisor, consequents_question_for
from habitus.scene.advice.prior import question_for as prior_question
from habitus.scene.advice.store import AdviceStore
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import comparable


class LanePriors:
    """``concepts`` 是这次认的概念集（换代折叠时是盘上的概念集；模型队列补问时是"概念集 + 同步词表将要生成的基础概念 +
    待生效的概念"）。``settled`` 是其中已经进了概念集的那些身份（不给就是全部）。"""

    def __init__(
        self,
        store: AdviceStore,
        advisor: PriorAdvisor,
        concepts: ConceptSet,
        lane: str,
        *,
        settled: Container[str] | None = None,
        zone: tzinfo | None = None,
    ) -> None:
        self.store, self.advisor, self.concepts, self.lane = store, advisor, concepts, lane
        self.settled = settled
        self.zone = zone if zone is not None else UTC
        self.names = [name for name in concepts.behaviors() if concepts.lane_of(name) == lane]
        self.ids = {name: canonical_digest(pair_part(concepts, name))[:16] for name in self.names}
        self._by_consequent: dict[str, dict[str, PriorLevel]] = {}
        self._by_antecedent: dict[str, dict[str, PriorLevel]] = {}

    # ── 缓存 ────────────────────────────────────────────────────────────────

    def consequent_key(self, consequent: str) -> str:
        return canonical_digest(
            {"lane": self.lane, "consequent": pair_part(self.concepts, consequent), "advisor": self.advisor.version}
        )

    def antecedent_key(self, antecedent: str) -> str:
        return canonical_digest(
            {
                "lane": self.lane,
                "antecedent": pair_part(self.concepts, antecedent),
                "advisor": self.advisor.version_by_antecedent,
            }
        )

    def _old(self, consequent: str) -> dict[str, PriorLevel]:
        if consequent not in self._by_consequent:
            self._by_consequent[consequent] = dict(self.store.prior(self.consequent_key(consequent)) or {})
        return self._by_consequent[consequent]

    def _new(self, antecedent: str) -> dict[str, PriorLevel]:
        if antecedent not in self._by_antecedent:
            found = self.store.prior_by_antecedent(self.antecedent_key(antecedent))
            self._by_antecedent[antecedent] = dict(found or {})
        return self._by_antecedent[antecedent]

    def level(self, antecedent: str, consequent: str) -> PriorLevel | None:
        """这一对的档：新问法那份里有就用它，否则用旧问法那份；都没有是 None。"""

        found = self._new(antecedent).get(self.ids[consequent])
        return found if found is not None else self._old(consequent).get(self.ids[antecedent])

    # ── 归题 ────────────────────────────────────────────────────────────────

    def seen(self, name: str) -> date | None:
        """这个概念出现的那一天（创建时刻的当地日期）；还没进概念集的是 None。"""

        if self.settled is not None and name not in self.settled:
            return None
        return self.concepts[name].created_at.astimezone(self.zone).date()

    def questions(self) -> Questions:
        """现在要问的题：还没有答案的对子按出现先后归到旧问法（按后果）或新问法（按前因）。"""

        return assign(
            self.names,
            comparable=lambda antecedent, consequent: comparable(self.concepts, antecedent, consequent),
            seen=self.seen,
            answered=lambda antecedent, consequent: self.level(antecedent, consequent) is not None,
        )

    def missing(self) -> int:
        """有几个后果还缺先验（它可比的前因里有没答案的）。"""

        return sum(
            1
            for consequent in self.names
            if any(
                comparable(self.concepts, antecedent, consequent) and self.level(antecedent, consequent) is None
                for antecedent in self.names
            )
        )

    def levels(self) -> dict[tuple[str, str], PriorLevel]:
        found: dict[tuple[str, str], PriorLevel] = {}
        for consequent in self.names:
            for antecedent in self.names:
                if comparable(self.concepts, antecedent, consequent):
                    level = self.level(antecedent, consequent)
                    if level is not None:
                        found[(antecedent, consequent)] = level
        return found

    # ── 问、清 ──────────────────────────────────────────────────────────────

    async def ask(self, signals: list[str]) -> None:
        """把现在要问的题都问了、写缓存：先旧问法（按后果），再新问法（按前因）。一题没答成，这一题这一回不写，下次再问。"""

        questions = self.questions()
        for consequent, antecedents in questions.by_consequent.items():
            answer = await self.advisor.rate(prior_question(self.concepts, self.lane, consequent, antecedents))
            signals.extend(answer.signals)
            if answer.levels:
                cached = self._old(consequent)
                cached.update({self.ids[antecedent]: level for antecedent, level in answer.levels.items()})
                self.store.write_prior(
                    self.consequent_key(consequent),
                    lane=self.lane,
                    consequent=consequent,
                    levels=cached,
                    version=answer.version,
                    models=answer.models,
                )
        for antecedent, consequents in questions.by_antecedent.items():
            answer = await self.advisor.rate_consequents(
                consequents_question_for(self.concepts, self.lane, antecedent, consequents)
            )
            signals.extend(answer.signals)
            if answer.levels:
                cached = self._new(antecedent)
                cached.update({self.ids[consequent]: level for consequent, level in answer.levels.items()})
                self.store.write_prior_by_antecedent(
                    self.antecedent_key(antecedent),
                    lane=self.lane,
                    antecedent=antecedent,
                    levels=cached,
                    version=answer.version,
                    models=answer.models,
                )

    def orphans(self) -> tuple[frozenset[str], frozenset[str]]:
        """这条 lane 没人再读的缓存键（旧问法的, 新问法的）：概念改名、改判据、删了，或换了先验触点的版本与模型。"""

        old = frozenset(self.store.prior_keys(self.lane)) - {self.consequent_key(name) for name in self.names}
        new = frozenset(self.store.prior_by_antecedent_keys(self.lane)) - {
            self.antecedent_key(name) for name in self.names
        }
        return old, new

    def discard_orphans(self) -> None:
        old, new = self.orphans()
        for key in old:
            self.store.discard_prior(key)
        for key in new:
            self.store.discard_prior_by_antecedent(key)


def pair_part(concepts: ConceptSet, name: str) -> list[str]:
    """先验缓存键里一个概念的那一份：给模型看的名字与判据（改名或改判据就是另一个问题）。"""

    return [concepts.label_of(name), concepts[name].definition]


__all__ = ["LanePriors", "pair_part"]
