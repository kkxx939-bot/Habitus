"""行为融合的提示词与输入渲染。

提示词只承载**局部可判的纯语义判断**；结构不变量由输出形状承担（``frames`` 逐帧归属表），我们
自己产物的自洽性由 ``validation`` 确定性校验。三者不能混。

## 示例的格式刻意与真实输入不同，不要"修正"

示例里的片段写成 ``#1 (+0s)  人走到水池边``，而模型实际收到的是
``#1 (+0s)  人走到水池边``。曾经认为这是缺陷并改成一致，实测
**status 类的稳定性从 5/5、5/5 掉到 5/5、3/5**（10 次对 10 次，不是噪音）。同一次改动里把观测
空白那一节写长，还让一条与空白毫无关系的因果用例从 3/3 掉到 1/3。

机制大概是：示例的作用是让模型看清**判断的结构**，把输入的字节复刻进去只会稀释教学信号；而
提示词是一个整体，任何一段变长都会挤压别处。所以这里不追求"示例与输入逐字一致"——要改先跑
``python -m benchmark.fusion`` 拿数字，不要凭"格式应该统一"这类说得通的理由动手。

## 要教一件事，就把它放进示例，光在正文里说不管用

"画面里可能不止一个人"这一节的正文写得很明确（别人的行为另起一条、不要塞进主体的判断），但两个
示例**都是单人的**，模型从没见过旁人判断长什么样——旁人那一帧被吸收进主体行为的比例是 2/3。往
示例一里加一帧"家庭成员B端着水杯走过"并单独成条之后，同一用例 8/8 稳定通过，全量回归无回退。

这和上一节是同一个机制的两面：示例是模型照抄的对象，正文只是背景。所以判断一段指令有没有生效，
看的是**它在示例里有没有对应的形态**，而不是正文写得够不够清楚。

## 说一件事有三种落点，效力依次递增

同一条规则放在哪里，效果差很远。按实测排序（弱 → 强）：

1. **正文段落**——最弱。"continues 不能指向 completed"写了一整段【重要】，模型仍然 **5/5** 先写错
   再被守卫打回，每次多烧两次调用（早上洗一次手、中午又洗一次，被判成第一次的后半段）。
2. **示例要点**——把同一句话加长 122 字写进示例二的要点，仍然 **5/5 无改善**。
3. **贴在模型正在看的那个对象上**——最强。把约束就地写进上下文那一行
   （``C1 …，completed/observed，已结束（不能被 continues 指）``），同一用例的结构重试**归零**，
   而真该延续的正例不受影响。

4. **schema 的字段描述**——与第 3 档同级，而且**不占提示词空间**。模型填某个字段时读的就是它。

所以"规则写了但模型不听"时，正确的下一步**不是把规则写得更长更重**，而是问：这条规则该贴在哪个
对象上？同理，``## basis`` 的小节标题曾经写成"只有 goal 非空时才填"——那是必要条件的说法，读起来
像"允许不填"，而代码当成充要。标题就是模型决定填不填时看的那个对象，话说反了比没说更糟。

提示词空间是真的稀缺：把四条硬契约（goal 非空必须有 basis、关系目标的合法性、subjects 不重复、
interrupted 与 abandoned 的区别）写进正文只净增 **65 字**（不到 1%），``status-abandoned`` 就从
**8/8 掉到 5/8**（同样本量头对头）；把同样四条搬进 schema 描述，契约照样在，提示词一字未增，
两条用例都回到 8/8。所以面向模型的契约优先写 schema，正文只留判断方法本身。

## 约束要贴在它约束的那个对象上，别贴在"要不要做"的位置

第一次搬进 schema 时，把"关系目标只能是别的、读得懂的判断"写在了 **relations 数组**的描述上：
"独立就填 []。可以有多条，**但**目标只能是……"。结果 ``concurrent-cook-and-call`` 从 8/8 掉到 7/8，
两次全量里更是 3/3 掉到 1/3——**模型漏标的正是它本来判得出的那条并行**。那个"但"是限制性从句，
摆在模型决定"要不要发关系"的位置上，把发关系这件事本身一起压住了。

这条约束其实是关于**选哪个目标**的，所以它属于 ``target`` 字段。搬过去之后立刻回到 8/8。

这与另一条老教训同源：给关系类型时用"判不出就不要标，留空比编一个更好"这种保守措辞，模型会**一个
都不标**，包括它明明判得出的。凡是出现在"要不要产出"位置上的限制、告诫、保守措辞，都要当成会压制
产出来对待；真正的限制要挪到"具体怎么填"的那个字段上。

还有一类根本不该靠提示词解决：模型在 ``judgements`` 与 ``frames`` 两张表之间的记账疏漏（声明了判断
却没给它任何帧、frames 标了一个没声明的 basis 编号）。它们是组合式记账不是语义判断，实测同一输入
三次里犯两次，提示词压不住。这类一律在装配层**降级**（丢掉那条、或退回判断级归属），而不是整批
拒绝——整批拒绝要白烧一次完整调用并吃掉一次重试预算。

一条纪律：提示词不去规定现实中的行为该长什么样。片段连不连续、一帧属不属于两件事，都是上游数据
的事实，不是本层能立的法。历史上被真实数据推翻的约束（一帧只能属于一处、全部片段必须无重无漏地
划分、动作的证据帧必须连续）无一例外都属于这一类。

判据的理论依据是活动理论的层级（由什么驱动）：由**有意识目标**驱动的是一件事，由条件驱动的是它
内部的操作。层级因此由"目标能不能再分解成多个独立目标"决定，而不是由时长决定。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from behavior.fusion.errors import BehaviorFusionError
from behavior.observation import BehaviorObservation

FUSION_PROMPT_VERSION = "behavior_judgement_prompt_v15"

FUSION_SYSTEM_PROMPT = """\
你在为一套行为记忆系统做行为融合。

输入是一段连续的行为语义片段。上游每隔几秒抽帧一次并转成语义，所以**相邻片段
经常在描述同一件正在进行的事**——把它们折叠起来是你最主要的工作，而不是切分。

每条片段写成 `#序号 (+偏移秒) [来源/可靠度] 参与者：语义`。可靠度三档：
`observed` 直接看到的、`inferred` 上游推断的、`reported` 别处报来的。
后两档比第一档弱，只靠它们支撑的判断要更保守——尤其别据此硬编目标。

## 你产出的是「判断」，不是「事实」

你不是在陈述一件已经发生完的事，而是在说：**看完这段观测，我判断这里发生了什么**。
判不准就说得少一点，不要硬编。

## 画面里可能不止一个人

user 消息开头会给出**本次跟踪的主体**。画面里出现别人是常态（家人、客人、保姆），他们的行为
**照常产出判断**，subjects 写那个人——**不要塞进主体的判断里**，那会让一件与主体无关的事成为
他行为的证据。系统会自己把不属于主体的判断分流出去，你只负责如实写谁在做。

反过来也不要漏：主体和别人**一起**做的事（两人合抬一张桌子），subjects 两个都写，那仍然是
主体的行为。

## 每条判断都要写的两个字段

  subjects  谁在做这件事，**数组**。必须逐字取自这条判断覆盖的片段里出现过的称呼，
            不要自己另起名字。**两个人一起做的事就都写上**——不要拼成一个字符串，
            也不要只留一个。behavior 为空时填 []。
  summary   一句话，脱离观测也读得懂。behavior 为空时填 null。

这两个和 status / status_basis 一样，**behavior 非空时一律必填**（subjects 至少一个人）。

## 三层，由两个字段决定

  behavior 有 + goal 有   →  一件有目标的事      「洗手」目标「清洁双手」
  behavior 有 + goal 空   →  只是一个动作        「打哈欠」
  behavior 空             →  这段观测没读懂      （其余字段全部留空）

判不出目标就**把 goal 留空**，不要硬编一个目标——那样产出的是假事实。
但反过来：能判出目标就必须填，留空不是躲避判断的地方。
连在做什么都看不出来（画面模糊、遮挡），就让 behavior 也留空。

## 输出结构：两段

  judgements   每条判断一项，写它的语义
  frames       **每条输入片段恰好一行**，写它属于哪条判断的哪个 basis

【最重要】frames 必须**逐行覆盖输入的每一条片段**，一行都不能少，顺序与输入完全
          一致，第 n 行的 no 必须等于 n。包括你觉得平淡无奇的过渡帧——"人走向厨房"
          "人经过客厅"这类位移帧最容易被漏掉，但它们同样服务于某个目标。

          每一行都必须至少有一个归属。读不懂的帧，归给一条 behavior 为空的判断。

一帧可以属于**多条判断**：一边吃饭一边看手机时，交界那一帧两边都成立就都写上。

## 什么是一件事

一件事对应主体的**一个有意识的目标**——他自己会说"我在做 X"的那个 X。

层级判据：**这个目标能不能再分解成多个各自独立的目标？**
  子部分脱离它就没有意义（洗手 → 打肥皂）           → 子部分写进 basis，它是一件事
  子部分能独立成立（回家 → 进门、换鞋、清洁、调温）  → 它太粗了，拆开

## basis：只有 goal 非空时才填

basis 是**构成这件事的行为事实**，是折叠之后的说法，不是原始帧的复述。
「洗手」的 basis 是「打开水龙头冲手」「打肥皂搓手」「冲水关龙头擦干」，
不是把八条"人在搓手"抄一遍。

goal 为空时 basis 填 []——那条判断本身就在动作这一层，里面没有东西可再分解。

## 折叠：你最常做的判断

对每条片段问：相对上一条，是**延续**（同一件事还在进行）、还是**换了目标**？
大多数是延续。不要因为措辞略有不同就判成新的一件事——
"人在搓手"和"人手上有泡沫在揉搓"是同一个动作的两次采样。

【重要】位置变了、对象变了，**都不直接构成切分理由**。它们只是提示你复核一次目标
        是否也变了。复核后目标未变就必须合并——做饭会在冰箱、水池、灶台之间移动
        并操作多种物品，那始终是**一件**事。

【重要】目标没有改变时**必须合并**。"我不确定"本身不是切分理由。

## status：这件事现在什么状态

  ongoing       还在做
  completed     做完了
  interrupted   被别的事打断，没回来（洗手洗到一半去接电话）
  abandoned     没人打断，自己不做了（衣物放进洗衣机又取出来抱走）

## status_basis：这个状态你是怎么知道的

  observed          看到了结束（关水龙头、甩手、擦手）
  inferred          没看到结束，但从后面推出来（后面在吃饭，说明做饭完了）
  observation_lost  观测在这里断了，之后发生什么不知道（人走出画面）

【重要】interrupted 和 observation_lost 完全不同，不要混。
        interrupted 是**你看到他没做完**；observation_lost 是**你没得看**——
        他可能做完了，也可能没有。分不清就用 observation_lost。

## relations：这条判断与别的判断的关系

独立的一件事就填 []。一条判断可以同时有多条关系。

  continues         延续另一条判断（它被切开了，这是后半段）
  supersedes        修正另一条判断（后面的观测推翻了先前的判断）
  concurrent_with   与另一条判断同时进行
  results_from      这件事**是那件事造成的结果**，或者那件事**使这件事成为可能或必要**

results_from 的例子：

  开空调 → 脱外套      室温升高了，所以脱外套
  烧水   → 泡茶        烧水是泡茶的必要前提
  洗手   → 徒手拿食物  手干净了，才徒手拿

仅仅时间上一前一后不构成因果。注意方向：由**结果**那条指回**原因**那条。

【重要】看不出因果就**不要标**，这不是失败。"这次做完饭接着吃饭"到底是不是因果，要看很多次
        才知道，那是系统自己会去统计的事，不需要你在这一次里判出来。你只要把看得清清楚楚的
        那种标出来（开空调→室温升高→脱外套），其余的如实留空——标出来的可信，比标得多重要。

目标有两个来源，**恰好填一个，另一个填 null**：

  target          本次输出里另一条判断的 judgement_no
  context_target  【先前的判断】里的编号（看到 C2 就填 2）

【重要】切段是机械的（到条数或时长上限就切），所以**一件事经常横跨两次融合**。上一段的后半截
        就在【先前的判断】里——看到它就用 context_target 指回去，不要当成全新的一件事。

【重要】continues 只能指向**还没做完**的那条（ongoing，或观测断了不知道做完没有）。
        已经 completed 的事没有"后半段"——同样的行为再做一次是**新的一件事**，不是延续。
        早上洗一次手、中午又洗一次，那是两次，不要连起来。
        如果先前那条判错了（写了 completed 其实没完），用 supersedes 修正它。
        同理，如果这一段的观测推翻了先前的判断（先前说"在水池边做什么看不出"，现在看清是洗手），
        用 supersedes 指回去，**不要重复描述**先前那条。

**并行只需要在一边声明**，系统会自动补上另一边——不要为了对称回头去改先前那条。
指向【先前的判断】的关系补不了对称（那条已经落盘、改不了），单向声明即可。

并行与中断的区别：**两个目标能否同时保持进行**。
  吃饭时玩手机——饭没停、人还在桌前   → 两条判断，concurrent_with
  洗手时接电话——手上是泡沫、人离开了 → 洗手那条 interrupted

【重要】这两者对同一对判断是**互斥**的：如果甲是被乙打断的，就只写甲 interrupted，
        **不要再给甲和乙标 concurrent_with**——一件被打断的事不可能同时还在与打断它的
        那件事并行。（甲被丙打断、同时与乙并行，是另一回事，那可以标。）

## 观测空白

片段之间的时间偏移会出现明显跳跃，这表示**这段时间我们没有观测**，不表示什么都没
发生——人可能走出了视野。

  - 空白**不构成**"这件事已经做完"的证据。没看到结束就用 observation_lost。
  - 空白后的片段**不要默认延续**空白前的行为，除非回归后的片段自身带有延续证据。

## 你不负责的事

不要输出任何时间、置信度、证据引用或标识符——它们由系统从片段确定性派生。
你只负责语义判断。

## 示例一：折叠、切分、旁人与读不懂

本次跟踪的主体：家庭成员A

输入片段：
  #1 (+0s)  人走到水池边
  #2 (+4s)  人站在水池边，手伸向水龙头
  #3 (+8s)  水流出，人手在水下
  #4 (+12s)  人手上有泡沫，在搓
  #5 (+16s)  人在搓手
  #6 (+20s)  人在搓手
  #7 (+22s)  家庭成员B端着水杯从画面右侧走过
  #8 (+24s)  水流，人在冲手
  #9 (+28s)  人关水龙头，手甩水
  #10 (+33s)  人打了个哈欠
  #11 (+37s)  画面模糊，无法判断
  #12 (+41s)  人拿起空调遥控器
  #13 (+45s)  空调启动，人放下遥控器


judgements：
  1  subjects=[家庭成员A]  behavior=洗手  goal=清洁双手  summary=回家后洗了手
     status=completed  status_basis=observed  relations=[]
     basis: 1「走到水池边打开水龙头」 2「打肥皂搓手」 3「冲水关龙头甩手」
  2  subjects=[家庭成员B]  behavior=端着水杯走过  goal=null  summary=另一个人端着水杯经过
     status=completed  status_basis=observed  relations=[]  basis=[]
  3  subjects=[家庭成员A]  behavior=打哈欠  goal=null  summary=打了一个哈欠
     status=completed  status_basis=observed  relations=[]  basis=[]
  4  behavior=null  subjects=[]  summary=null  goal=null
     status=null  status_basis=null  relations=[]  basis=[]      ← 这段没读懂
  5  subjects=[家庭成员A]  behavior=开空调  goal=调节室温  summary=打开了空调
     status=completed  status_basis=observed  relations=[]
     basis: 1「拿起遥控器开机」

frames：
  no=1 [(1,1)]  no=2 [(1,1)]  no=3 [(1,1)]
  no=4 [(1,2)]  no=5 [(1,2)]  no=6 [(1,2)]
  no=7 [(2,null)]
  no=8 [(1,3)]  no=9 [(1,3)]
  no=11 [(4,null)]
  no=12 [(5,1)]  no=13 [(5,1)]

要点：
  frames 一共 13 行，对应 13 条输入片段，一行不多一行不少。
  #1 是位移，但它服务于洗手，归进 basis 1，**不要漏掉它**。
  #5 #6 不是两个动作，是同一动作被采样两次，都归 basis 2。
  #7 是**别人**在做的事，另起一条判断、subjects 写家庭成员B，**不要**因为它夹在洗手中间
      就归进第 1 条——旁人经过不是主体洗手的证据。第 1 条因此覆盖 1-6 和 8-9，中间断开
      是正常的。
  #10 打哈欠判不出目标，goal 留空，basis 填 []——它本身就是一个动作。
  #11 读不懂，单独一条 behavior 为空的判断，**不要丢弃**。

## 示例二：并行与观测空白

输入片段：
  #1 (+0s)  人坐在餐桌前，面前有食物
  #2 (+5s)  人夹菜送入口中
  #3 (+10s)  人拿起手机
  #4 (+15s)  人看手机屏幕，另一只手拿筷子
  #5 (+20s)  人在看手机
  #6 (+25s)  人夹菜
  #7 (+30s)  人放下筷子，起身走向画面外
  #8 (+295s)  人从画面外回来
  #9 (+300s)  人坐回餐桌继续吃


judgements：
  1  subjects=[家庭成员A]  behavior=吃饭  goal=吃完这顿饭  summary=在餐桌前吃饭，中途离开
     status=ongoing  status_basis=observation_lost
     relations=[]                          ← 并行由第 2 条声明就够了
     basis: 1「在餐桌前进食」
  2  subjects=[家庭成员A]  behavior=看手机  goal=查看手机内容  summary=边吃饭边看手机
     status=completed  status_basis=observed
     relations=[{concurrent_with, 1}]
     basis: 1「拿起手机注视屏幕」
  3  subjects=[家庭成员A]  behavior=离开餐桌  goal=null  summary=放下筷子起身走开
     status=completed  status_basis=observed  relations=[]   basis=[]
  4  subjects=[家庭成员A]  behavior=继续吃饭  goal=吃完这顿饭  summary=回到餐桌继续吃
     status=ongoing  status_basis=observation_lost
     relations=[{continues, 1}]
     basis: 1「回到餐桌继续进食」

frames：
  no=1 [(1,1)]  no=2 [(1,1)]
  no=3 [(2,1)]  no=4 [(1,1),(2,1)]  no=5 [(2,1)]
  no=6 [(1,1)]
  no=7 [(3,null)]
  no=8 [(4,1)]  no=9 [(4,1)]

要点：
  #4「人看手机屏幕，另一只手拿筷子」两件事都在做，所以归给两条判断。
  吃饭覆盖的片段 1,2,4,6 是**不连续**的，这是并行的正常形态，不要因此判成中断。
  #7 起身走开——判不出他去做什么，goal 留空。
  空白之后 #8 #9 带有明确的延续证据（坐回餐桌继续吃），所以用 continues 指回第 1 条；
  如果回来之后做的是别的事，relations 就留空。

## 示例三：转写、途中小动作与边做边聊

本次跟踪的主体：成年女性A

输入片段：
  #1 (+0s)  成年女性A打开冰箱拿出鸡蛋
  #2 (+6s)  成年女性A把鸡蛋磕进碗里
  #3 (+9s) [audio/reported] 少年B：妈，今晚吃什么？
  #4 (+12s)  成年女性A把蛋壳扔进垃圾桶
  #5 (+15s) [audio/reported] 成年女性A：煎蛋饼，再炒个青菜。
  #6 (+20s)  成年女性A打开水龙头冲洗双手并擦干
  #7 (+26s) [audio/reported] 少年B：那我作业写完能看会儿电视吗？
  #8 (+31s) [audio/reported] 成年女性A：写完先给我检查。
  #9 (+36s)  成年女性A搅打碗里的鸡蛋


judgements：
  1  subjects=[成年女性A]  behavior=做晚饭  goal=准备晚饭  summary=一边和孩子说话一边准备晚饭
     status=ongoing  status_basis=observation_lost  relations=[]
     basis: 1「取出鸡蛋磕进碗里，蛋壳扔进垃圾桶」 2「中途冲洗双手擦干」 3「搅打蛋液」
  2  subjects=[成年女性A、少年B]  behavior=与少年B交谈  goal=null
     summary=做饭时回答孩子晚饭吃什么、写完作业能不能看电视
     status=completed  status_basis=observed
     relations=[{concurrent_with, 1}]  basis=[]

frames：
  no=1 [(1,1)]  no=2 [(1,1)]
  no=3 [(2,null)]
  no=4 [(1,1)]
  no=5 [(2,null)]
  no=6 [(1,2)]
  no=7 [(2,null)]  no=8 [(2,null)]
  no=9 [(1,3)]

要点：
  [audio/reported] 的转写行与画面行是同一条时间线上的观测，同等参与判断。
  #4 扔蛋壳、#6 中途洗手是做饭的**服务性动作**，归进 basis，不另立判断——
      与示例一"回家后洗手"不同：那次洗手自成目标，这次只是做饭途中的一步。
  四句对话跨了两个话题（晚饭、看电视），仍是**一次交谈**，折成一条，不按话题切开。
  边做饭边聊天是并行的两件事，concurrent_with 在一边声明即可。
  孩子的问话是这场交谈的一部分，subjects 两个都写；他**单独**做的事才另起判断。
"""


def render_primary_subject(primary_subject: str) -> str:
    """把"本次跟踪谁"渲染给模型。

    这是**配置事实**不是语义判断：系统知道它在为谁服务，模型不该猜。上游保证 ``participants``
    里用的是跨批次稳定的标识，所以这里逐字给出即可。
    """

    if not isinstance(primary_subject, str) or not primary_subject.strip():
        raise BehaviorFusionError("primary_subject must be non-empty text")
    return primary_subject.strip()


def render_fragments(fragments: Sequence[BehaviorObservation]) -> str:
    """把观测片段渲染成带临时编号与相对秒偏移的清单。

    只给相对偏移而不给绝对时间：偏移足以让模型看出连续性与观测空白，而绝对时间会诱使它输出时间
    字段——那些字段一律由系统派生。编号与 ``frames`` 表使用的 ``1..N`` 一致。
    """

    if isinstance(fragments, (str, bytes)) or not isinstance(fragments, Sequence) or not fragments:
        raise BehaviorFusionError("fusion requires a non-empty fragment sequence")
    if any(not isinstance(item, BehaviorObservation) for item in fragments):
        raise TypeError("fragments must contain BehaviorObservation values")
    origin = fragments[0].occurred_at
    lines: list[str] = []
    for index, fragment in enumerate(fragments, start=1):
        offset = int((fragment.occurred_at - origin).total_seconds())
        participants = "、".join(fragment.participants)
        lines.append(
            f"#{index} (+{offset}s) [{fragment.modality.value}/{fragment.knowledge_state}] "
            f"{participants}：{fragment.semantics}"
        )
    return "\n".join(lines)


def render_context_judgements(
    records: Sequence[Mapping[str, Any]],
    segment_started_at: datetime,
) -> str:
    """把先前已经落盘的判断渲染成融合上下文。

    编号 ``C1..Cn`` 就是 ``context_target`` 要填的数字——模型不可能写对 64 位的内容身份，所以
    这里给它小整数，由系统映射回真实身份。

    只给相对本段起点的偏移而不给绝对时间，与片段渲染一致：模型不负责输出任何时间字段。
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise BehaviorFusionError("context judgements must be a sequence")
    if not records:
        return "（无）"
    if not isinstance(segment_started_at, datetime) or segment_started_at.utcoffset() is None:
        raise TypeError("segment_started_at must be a timezone-aware datetime")
    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        started_at = datetime.fromisoformat(str(record["started_at"]))
        ago = int((segment_started_at - started_at).total_seconds())
        behavior = record.get("behavior") or "（没读懂）"
        goal = record.get("goal")
        status = record.get("status")
        basis = record.get("status_basis")
        state = "" if status is None else f"，{status}/{basis}"
        purpose = "" if goal is None else f"，目标：{goal}"
        # 约束贴在它作用的那一行上，而不是放进一段远处的说明。实测：提示词正文里已经有一整段
        # 讲"continues 不能指向 completed"，模型仍然 5/5 全部先写错再被守卫打回（每次多烧两次
        # 调用）；而把同一句话加长写进要点，仍然 5/5 无改善。模型看的是 C 行本身。
        closed = "，已结束（不能被 continues 指）" if status == "completed" else ""
        lines.append(f"C{index}  {ago}秒前开始：{behavior}{purpose}{state}{closed}")
    return "\n".join(lines)


__all__ = [
    "FUSION_PROMPT_VERSION",
    "FUSION_SYSTEM_PROMPT",
    "render_context_judgements",
    "render_fragments",
]
