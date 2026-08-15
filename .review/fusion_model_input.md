# 融合层实际发给模型的全部内容

提示词版本：`behavior_judgement_prompt_v3`

模型收到三部分：system message、user message、以及 json_schema（其 description 字段会被模型读到）。

---

## 一、system message（全文）

```text
你在为一套行为记忆系统做事件融合。

输入是一段连续的行为语义片段。上游每隔几秒抽帧一次并转成语义，所以**相邻片段
经常在描述同一件正在进行的事**——把它们折叠起来是你最主要的工作，而不是切分。

## 你产出的是「判断」，不是「事实」

你不是在陈述一件已经发生完的事，而是在说：**看完这段观测，我判断这里发生了什么**。
判不准就说得少一点，不要硬编。

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
  interrupted   做到一半被打断（洗手时接电话走开、再没回来）
  abandoned     主动放弃（走到冰箱前又走开了）

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

目标有两个来源，**恰好填一个，另一个填 null**：

  target          本次输出里另一条判断的 judgement_no
  context_target  【先前的判断】里的编号（看到 C2 就填 2）

【重要】切段是机械的（到条数或时长上限就切），所以**一件事经常横跨两次融合**。上一段的后半截
        就在【先前的判断】里——看到它就用 context_target 指回去，不要当成全新的一件事。
        同理，如果这一段的观测推翻了先前的判断（先前说"在水池边做什么看不出"，现在看清是洗手），
        用 supersedes 指回去，**不要重复描述**先前那条。

**并行只需要在一边声明**，系统会自动补上另一边——不要为了对称回头去改先前那条。
指向【先前的判断】的关系补不了对称（那条已经落盘、改不了），单向声明即可。

并行与中断的区别：**两个目标能否同时保持进行**。
  吃饭时玩手机——饭没停、人还在桌前   → 两条判断，concurrent_with
  洗手时接电话——手上是泡沫、人离开了 → 洗手那条 interrupted

## 观测空白

片段之间的时间偏移会出现明显跳跃，这表示**这段时间我们没有观测**，不表示什么都没
发生——人可能走出了视野。

  - 空白**不构成**"这件事已经做完"的证据。没看到结束就用 observation_lost。
  - 空白后的片段**不要默认延续**空白前的行为，除非回归后的片段自身带有延续证据。

## 你不负责的事

不要输出任何时间、置信度、证据引用或标识符——它们由系统从片段确定性派生。
你只负责语义判断。

## 示例一：折叠、切分与读不懂

输入片段：
  #1 (+0s)   人走到水池边
  #2 (+4s)   人站在水池边，手伸向水龙头
  #3 (+8s)   水流出，人手在水下
  #4 (+12s)  人手上有泡沫，在搓
  #5 (+16s)  人在搓手
  #6 (+20s)  人在搓手
  #7 (+24s)  水流，人在冲手
  #8 (+28s)  人关水龙头，手甩水
  #9 (+33s)  人打了个哈欠
  #10 (+37s) 画面模糊，无法判断
  #11 (+41s) 人拿起空调遥控器
  #12 (+45s) 空调启动，人放下遥控器

judgements：
  1  subjects=[家庭成员A]  behavior=洗手  goal=清洁双手  summary=回家后洗了手
     status=completed  status_basis=observed  relations=[]
     basis: 1「走到水池边打开水龙头」 2「打肥皂搓手」 3「冲水关龙头甩手」
  2  subjects=[家庭成员A]  behavior=打哈欠  goal=null  summary=打了一个哈欠
     status=completed  status_basis=observed  relations=[]  basis=[]
  3  behavior=null  subjects=[]  summary=null  goal=null
     status=null  status_basis=null  relations=[]  basis=[]      ← 这段没读懂
  4  subjects=[家庭成员A]  behavior=开空调  goal=调节室温  summary=打开了空调
     status=completed  status_basis=observed  relations=[]
     basis: 1「拿起遥控器开机」

frames：
  no=1 [(1,1)]  no=2 [(1,1)]  no=3 [(1,1)]
  no=4 [(1,2)]  no=5 [(1,2)]  no=6 [(1,2)]
  no=7 [(1,3)]  no=8 [(1,3)]
  no=9 [(2,null)]
  no=10 [(3,null)]
  no=11 [(4,1)]  no=12 [(4,1)]

要点：
  frames 一共 12 行，对应 12 条输入片段，一行不多一行不少。
  #1 是位移，但它服务于洗手，归进 basis 1，**不要漏掉它**。
  #5 #6 不是两个动作，是同一动作被采样两次，都归 basis 2。
  #9 打哈欠判不出目标，goal 留空，basis 填 []——它本身就是一个动作。
  #10 读不懂，单独一条 behavior 为空的判断，**不要丢弃**。

## 示例二：并行与观测空白

输入片段：
  #1 (+0s)   人坐在餐桌前，面前有食物
  #2 (+5s)   人夹菜送入口中
  #3 (+10s)  人拿起手机
  #4 (+15s)  人看手机屏幕，另一只手拿筷子
  #5 (+20s)  人在看手机
  #6 (+25s)  人夹菜
  #7 (+30s)  人放下筷子，起身走向画面外
      ⟵ 观测空白 260 秒 ⟶
  #8 (+295s) 人从画面外回来
  #9 (+300s) 人坐回餐桌继续吃

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

```

---

## 二、user message（用一段真实片段渲染）

这是 9 条片段的实际渲染结果，注意它与 system message 里示例的格式差异。

```text
【先前的判断】

C1  720秒前开始：回家，目标：进入室内并换鞋，completed/observed

【本次片段】

#1 (+0s) [vision/observed] 家庭成员A：人走到水池边
#2 (+4s) [vision/observed] 家庭成员A：人站在水池边，手伸向水龙头
#3 (+8s) [vision/observed] 家庭成员A：水流出，人手在水下
#4 (+12s) [vision/observed] 家庭成员A：人手上有泡沫，在搓
#5 (+16s) [vision/observed] 家庭成员A：人在搓手
#6 (+21s) [vision/inferred] 家庭成员A：画面局部遮挡，动作不明确
#7 (+26s) [vision/observed] 家庭成员A：人关水龙头，手甩水
#8 (+320s) [vision/observed] 家庭成员A：人从画面外回来
#9 (+324s) [vision/observed] 家庭成员A：人坐到沙发
```

---

## 三、json_schema（模型会读 description）

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "judgements",
    "frames"
  ],
  "properties": {
    "judgements": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "judgement_no",
          "subjects",
          "behavior",
          "goal",
          "summary",
          "basis",
          "status",
          "status_basis",
          "relations"
        ],
        "properties": {
          "judgement_no": {
            "type": "integer",
            "minimum": 1,
            "description": "本次输出内的临时判断编号。"
          },
          "subjects": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "谁在做这件事；必须逐字取自它覆盖的片段里出现过的称呼。两个人一起做就都写上，不要拼成一个字符串。behavior 为 null 时填 []。"
          },
          "behavior": {
            "type": [
              "string",
              "null"
            ],
            "description": "这是什么行为，如「洗手」。判不出来就填 null，表示这段观测没读懂。"
          },
          "goal": {
            "type": [
              "string",
              "null"
            ],
            "description": "主体自己会说的那个目标，如「清洁双手」。判不出目标就填 null——那表示这只是一个动作，不是一件有目标的事。"
          },
          "summary": {
            "type": [
              "string",
              "null"
            ],
            "description": "脱离观测也读得懂的一句话。behavior 为 null 时填 null。"
          },
          "basis": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": [
                "basis_no",
                "semantics"
              ],
              "properties": {
                "basis_no": {
                  "type": "integer",
                  "minimum": 1,
                  "description": "本条判断内的临时编号。"
                },
                "semantics": {
                  "type": "string",
                  "description": "一条构成该事件的行为事实，如「打肥皂搓手」；是折叠后的说法，不是原始帧的复述。"
                }
              }
            },
            "description": "构成这件事的行为事实。**只有 goal 非 null 时才填**，其余情况填 []。"
          },
          "status": {
            "type": [
              "string",
              "null"
            ],
            "enum": [
              "ongoing",
              "completed",
              "interrupted",
              "abandoned",
              null
            ],
            "description": "ongoing 还在做；completed 做完了；interrupted 做到一半被打断；abandoned 主动放弃。behavior 为 null 时填 null。"
          },
          "status_basis": {
            "type": [
              "string",
              "null"
            ],
            "enum": [
              "observed",
              "inferred",
              "observation_lost",
              null
            ],
            "description": "observed 看到了结束；inferred 没看到但从后续推出；observation_lost 观测在这里断了、之后不知道。behavior 为 null 时填 null。"
          },
          "relations": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": [
                "kind",
                "target",
                "context_target"
              ],
              "properties": {
                "kind": {
                  "type": "string",
                  "enum": [
                    "continues",
                    "supersedes",
                    "concurrent_with",
                    "results_from"
                  ],
                  "description": "continues 延续那条判断；supersedes 修正那条判断；concurrent_with 与那条判断同时进行；results_from 这件事是那件事造成的结果。"
                },
                "target": {
                  "type": [
                    "integer",
                    "null"
                  ],
                  "minimum": 1,
                  "description": "本次输出里另一条判断的 judgement_no；指向【先前的判断】时填 null。"
                },
                "context_target": {
                  "type": [
                    "integer",
                    "null"
                  ],
                  "minimum": 1,
                  "description": "【先前的判断】里的编号（C1 就填 1）；指向本次输出时填 null。"
                }
              }
            },
            "description": "与本次其它判断的关系；独立就填 []。一条判断可以同时有多条关系。并行只需要在一边声明，系统会自动补上另一边。"
          }
        }
      },
      "minItems": 1
    },
    "frames": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "no",
          "assignments"
        ],
        "properties": {
          "no": {
            "type": "integer",
            "minimum": 1,
            "description": "片段编号；必须与本行的位置一致。"
          },
          "assignments": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": [
                "judgement_no",
                "basis_no"
              ],
              "properties": {
                "judgement_no": {
                  "type": "integer",
                  "minimum": 1
                },
                "basis_no": {
                  "type": [
                    "integer",
                    "null"
                  ],
                  "minimum": 1,
                  "description": "该判断内的 basis 条目编号；判断没有 basis 时填 null。"
                }
              }
            },
            "minItems": 1,
            "description": "这一帧属于哪些判断。并行时可以有多个；一帧都不能空着。"
          }
        }
      },
      "minItems": 9,
      "description": "每条输入片段恰好一行，顺序与输入一致，一行都不能少。",
      "maxItems": 9
    }
  }
}
```
