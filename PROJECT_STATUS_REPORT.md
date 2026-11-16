# Taskove (Gillo) AI Service: Comprehensive Status Report
# تقرير شامل عن حالة خدمة الذكاء الاصطناعي (ما تم إنجازه وما لم يُنجز بعد)

**Date / التاريخ:** November 2026 / نوفمبر 2026  
**System / النظام:** Taskove (Gillo) AI Service  
**Test Suite / الاختبارات:** 60 Passing Tests / 60 اختباراً ناجحاً (81% Code Coverage)  
**Models Active / النماذج النشطة:**
- Model 1 (Story Points): `model_sp_taskove` (`v2_taskove`)
- Model 2 (Sprint Carry-Over Risk): `model2_sprint_carryover_risk_taskove` (`v2_taskove_features`)

---

## Table of Contents / جدول المحتويات
1. [Executive Summary / الملخص التنفيذي](#1-executive-summary--الملخص-التنفيذي)
2. [Model 1: Story Point Estimation / النموذج الأول: تقدير نقاط القصة](#2-model-1-story-point-estimation--النموذج-الأول-تقدير-نقاط-القصة)
3. [Model 2: Sprint Carry-Over Risk Scoring / النموذج الثاني: التنبؤ بمخاطر تعثر السبرنت](#3-model-2-sprint-carry-over-risk-scoring--النموذج-الثاني-التنبؤ-بمخاطر-تعثر-السبرنت)
4. [Suggestions Inbox & State Machine / صندوق الاقتراحات الذكية ودورة اتخاذ القرار](#4-suggestions-inbox--state-machine--صندوق-الاقتراحات-الذكية-ودورة-اتخاذ-القرار)
5. [Sprint Capacity & Fit Check / فحص سعة السبرنت وملاءمة العبء](#5-sprint-capacity--fit-check--فحص-سعة-السبرنت-وملاءمة-العبء)
6. [Agile Drafts & Generative LLMs / مسودات التقارير والذكاء التوليدي](#6-agile-drafts--generative-llms--مسودات-التقارير-والذكاء-التوليدي)
7. [Chat Assistant ("Ask the Project") / المساعد الحواري الذكي للمشروع](#7-chat-assistant-ask-the-project--المساعد-الحواري-الذكي-للمشروع)
8. [Architecture, Security & Telemetry / البنية التحتية، الأمان وسجلات الأداء](#8-architecture-security--telemetry--البنية-التحتية-الأمان-وسجلات-الأداء)
9. [Consolidated Scorecard / جدول التقييم والجاهزية الشامل](#9-consolidated-scorecard--جدول-التقييم-والجاهزية-الشامل)
10. [Data Architecture, Real Samples & Data Gaps / هيكلية البيانات، عينات تفصيلية، وما ينقص البيانات فعلياً](#10-data-architecture-real-samples--data-gaps--هيكلية-البيانات-عينات-تفصيلية-وما-ينقص-البيانات-فعليا)

---

## 1. Executive Summary / الملخص التنفيذي

### English
This document provides a complete, honest, and auditable inventory of everything built, hardened, and verified in the Taskove AI Service, alongside everything that remains open, unvalidated on production data, or planned for future development phases. The service has transitioned from an unreliable synthetic prototype with fabricated telemetry into an audited, production-grade microservice powered by empirical machine learning pipelines trained on the open-source TAWOS agile dataset (over 500,000 Jira issues and 92,000 sprint tasks).

### العربية
تُقدّم هذه الوثيقة حصراً شاملاً، دقيقاً، وموثقاً لكل ما تم بناؤه واختباره وتأمينه في خدمة Taskove AI، جنباً إلى جنب مع كل ما لم يُنجز بعد، وما لم يتم اختباره على بيانات الإنتاج الحقيقية، والمهام المؤجلة للمراحل القادمة. لقد تحولت الخدمة بالكامل من نموذج تجريبي هش يعتمد على بيانات وهمية وأرقام مصطنعة إلى خدمة إنتاجية متكاملة مدعومة بنماذج تعلّم آلي حقيقية مدربة على مجموعة بيانات TAWOS العالمية (أكثر من 500,000 مهمة وأكثر من 92,000 سبرنت من مشاريع برمجية مفتوحة المصدر).

---

## 2. Model 1: Story Point Estimation / النموذج الأول: تقدير نقاط القصة

### Overview / نظرة عامة
- **Task / المهمة:** Estimating agile story points from task title, description, and task type.
- **Active Version / الإصدار النشط:** `v2_taskove` (LightGBM L1, TF-IDF + Type, Quantiles Q25/Q50/Q75, No Priority).
- **Artifact Path / مسار الحزم:** `benchmark_v2/artifacts/model1_sp/v2_taskove/`

### A. What is DONE / ما تم إنجازه بالكامل

#### English
1. **Empirical Training on TAWOS:** Trained on 29,330 historical issues and evaluated on 6,167 test issues (chronological split, Epics excluded by product scope).
2. **Priority Exclusion:** Removed `Priority` from the feature set to ensure Taskove compatibility and avoid cross-tenant fragility (v1_tawos permanently superseded).
3. **Out-of-Project Conformal Calibration:** Calibrated the uncertainty interval constant ($c_{\text{cold-start}} = 0.1147$) using 41,598 pooled out-of-project residuals across 5 cross-validation folds, guaranteeing honest 50% intervals covering ~50.8% of unseen tickets with ~2.89 SP width.
4. **Snapping to Fibonacci:** Continuous regression outputs snap to the standard agile Fibonacci scale $\{0.5, 1, 2, 3, 5, 8, 13, 21\}$.
5. **Fair Baseline Comparison:** Evaluated model vs. per-type median baseline under identical treatments (Model MAE 1.7411, 63.42% within 1 point vs. Baseline 1.9729, 51.32% within 1 point; +12.10% advantage).
6. **Losing Projects Audit:** Debunked prior false claims; proved that the 9 losing cold-start projects all have $N \ge 134$ where static medians naturally excel due to narrow task distributions.
7. **Canonical Type Mapping Table:** Implemented mapping: `feature` $\to$ `Story`, `bug` $\to$ `Bug`, `docs` $\to$ `Documentation`, `refactor` $\to$ `Improvement`, `infra/test/backend/frontend/qa` $\to$ `Technical task`, `ml` $\to$ `Task`, unknown $\to$ `Task` (sensitivity impact < 0.017 MAE).
8. **Language Guard:** Evaluates input text for Latin character density; if $>30\%$ non-ASCII letters (e.g. Arabic or Cyrillic), falls back to per-type median with `fallback_reason="non_english_text"`.
9. **Contract Enforcement:** Strict verification of ordered input fields `['title', 'description', 'type']`. Mutation immediately falls back.
10. **Multi-Artifact Integrity:** Checks SHA-256 hashes of all 7 artifacts on startup.
11. **Zero HTTP 500 Guarantee:** Any inference exception, missing artifact, or contract failure gracefully falls back to `type_median_fallback.json` with an explicit method label.
12. **No Hours Conversion Policy:** Completely eliminated linear story-point-to-hours conversion in suggestions; labeled legacy hours as an unvalidated heuristic ($r \sim 0.06 - 0.30$).
13. **Parity Verification:** Verified 250 issues from the test partition bit-for-bit against benchmark artifacts ($|\Delta| < 10^{-6}$).

#### العربية
1. **تدريب حقيقي على TAWOS:** تدريب النموذج على 29,330 مهمة واختباره على 6,167 مهمة زمنياً (مع استبعاد الـ Epics بقرار نطاق المنتج لأن Taskove يُقدّر مهام السبرنت وليس الملاحم الكبرى).
2. **استبعاد الأولوية (Priority):** حذف حقل الأولوية نهائياً لعدم توفره في Taskove وتفادي هشاشة النقل بين الشركات (تم إلغاء v1_tawos نهائياً).
3. **معايرة فترات الثقة خارج نطاق المشاريع ($c_{\text{cold-start}}$):** احتساب ثابت المعايرة ($c=0.1147$) من 41,598 خطأ متبقياً عبر 5 طيات تحقق لمشاريع لم يرها النموذج، محققاً تغطية حقيقية بنسبة ~50.8% ومتوسط عرض 2.89 نقطة.
4. **المطابقة مع مقياس فيبوناتشي:** تقريب القيم المتصلة لأقرب رقم في مقياس أجايل القياسي $\{0.5, 1, 2, 3, 5, 8, 13, 21\}$.
5. **مقارنة عادلة مع الأساس:** مقارنة النموذج مع خط الأساس لمتوسط النوع تحت نفس المعاملة (النموذج حقق MAE 1.7411 ودقة 63.42% ضمن نقطة واحدة مقابل 1.9729 و51.32% للأساس).
6. **تدقيق المشاريع الخاسرة:** تصحيح الخطأ السابق وإثبات أن المشاريع التسعة التي تفوق فيها الأساس كلها مشاريع كبيرة ($N \ge 134$) تمتاز بتشتت منخفض للمهام.
7. **جدول مطابقة أنواع المهام:** مطابقة أنواع مهام Taskove مع أنواع TAWOS المعتمدة (`feature` إلى `Story`، `bug` إلى `Bug`، والأنواع المجهولة إلى `Task`).
8. **حارس اللغة (Language Guard):** فحص نسبة الحروف اللاتينية؛ إذا زادت الحروف غير اللاتينية عن 30% (مثل النصوص العربية)، يعود النظام تلقائياً للوسيط الحسابي مع توضيح السبب `non_english_text`.
9. **حماية عقد المدخلات:** التحقق الصارم من ترتيب الحقول `['title', 'description', 'type']`، وأي تغيير يعيد خط الأساس تلقائياً.
10. **سلامة الحزم البرمجية:** فحص بصمة SHA-256 لجميع ملفات النموذج السبعة عند بدء التشغيل.
11. **ضمان عدم حدوث خطأ HTTP 500:** أي استثناء أو ملف مفقود يعود تلقائياً لملف `type_median_fallback.json` مع بيان نوع الطريقة.
12. **حظر تحويل النقاط لساعات:** منع تحويل نقاط القصة لساعات في الاقتراحات؛ وتصنيف أي ساعات سابقة كمعادلة تقديرية غير مثبتة تجريبياً ($r \sim 0.06 - 0.30$).
13. **اختبار التطابق التام:** فحص 250 مسألة حقيقية من بيانات الاختبار وإثبات تطابق التنبؤات حتى 6 خانات عشرية ($< 10^{-6}$).

---

### B. What is NOT DONE & Future Scope / ما لم يتم إنجازه والمهام المستقبلية

#### English
1. **Not Validated on Taskove Production Data:** The model was trained entirely on open-source Jira repositories (TAWOS). Real distribution drift across private Taskove customer projects has not yet been measured.
2. **Backend Task Type Confirmation Pending:** A formal confirmation of the full production enum of task types from the backend team is marked as a `TODO` in `app/ml/taskove_types.py`.
3. **No Non-English Text Support:** The TF-IDF vocabulary is 100% English. Tasks written in Arabic, German, Spanish, etc., are redirected to median fallbacks and cannot leverage natural language features.
4. **No Real-Time Online Retraining:** The model is static; it does not currently adapt or learn on-the-fly when developers adjust or override story points in sprint planning.

#### العربية
1. **لم يتم التحقق على بيانات الإنتاج لـ Taskove:** تم تدريب النموذج بالكامل على مستودعات مفتوحة المصدر (TAWOS)؛ ولم يتم بعد قياس انحراف البيانات على مشاريع عملاء Taskove الفعليين.
2. **تأكيد أنواع المهام من فريق الباك إند:** يلزم اعتماد القائمة النهائية لأنواع المهام المستخدمة رسمياً في Taskove (تم وضع علامة `TODO` بوضوح في الكود).
3. **عدم دعم اللغات غير الإنجليزية:** معجم TF-IDF يعتمد كلياً على الإنجليزية؛ المهام المكتوبة بالعربية أو غيرها تُحال لخط الأساس للنوع ولا تستفيد من تحليل النص.
4. **عدم وجود إعادة تدريب لحظي (Online Learning):** النموذج ثابت ولا يتعلم تلقائياً من التعديلات التي يُجريها المستخدمون على نقاط القصة أثناء التخطيط.

---

## 3. Model 2: Sprint Carry-Over Risk Scoring / النموذج الثاني: التنبؤ بمخاطر تعثر السبرنت

### Overview / نظرة عامة
- **Task / المهمة:** Predicting the risk of sprint tasks failing to reach "Done" before sprint close.
- **Active Version / الإصدار النشط:** `v2_taskove_features` (LightGBM Classifier + Isotonic Calibrator + Feature Transformer).
- **Artifact Path / مسار الحزم:** `benchmark_v2/artifacts/model2/v2_taskove_features/`

### A. What is DONE / ما تم إنجازه بالكامل

#### English
1. **Full-Universe TAWOS Training:** Trained on 92,547 completed sprint tasks across diverse projects with strict temporal feature isolation ($f \in [0.2, 0.8]$).
2. **23-Feature Canonical Contract:** Enforced 23 exact features documented in `FEATURE_CONTRACT.md` (exact sprint assignment flags, backlog age, blocked status, etc.).
3. **Calibrated Probabilities:** Integrated Isotonic Regression calibrator to turn raw tree scores into true empirical probabilities.
4. **Telemetry Privacy Isolation:** Raw probabilities are strictly reserved for `ai_runs` internal telemetry and **never** exposed in client JSON responses.
5. **Top-$k$ Focus Policy:** Implemented localized within-sprint ranking where $k = \min(5, \lceil 0.2 \times n_{\text{open}} \rceil)$, grouping tasks into High, Medium, and Low risk tiers with top 3 contributing factors.
6. **Honest Rule Baseline Fallback:** If sprint is ineligible (inactive, $f \notin [0.2, 0.8]$, open tasks < 10, missing timestamps), falls back to a transparent heuristic: `todo` status first, stalled days second, size third.
7. **Daily Automated Risk Scanner Job:** APScheduler job automatically scans active sprints daily and deduplicates suggestions against existing pending or dismissed items.
8. **Parity Tested:** 200-row numerical parity verified within $10^{-5}$ tolerance.

#### العربية
1. **تدريب شامل على TAWOS:** تدريب النموذج على 92,547 مهمة سبرنت عبر مشاريع متعددة مع عزل زمني صارم للخصائص ($f \in [0.2, 0.8]$).
2. **عقد الخصائص الـ 23:** الالتزام الصارم بـ 23 خاصية محددة بدقة في وثيقة `FEATURE_CONTRACT.md` (تواريخ الإسناد الدقيقة، عمر المهمة، حالات التوقف، إلخ).
3. **معايرة الاحتمالات (Isotonic Calibration):** استخدام معاير متساوي التوتر لضمان تعبير الاحتمالات عن النسب الحقيقية لتعثر المهام.
4. **عزل الاحتمالات الرقمية عن العميل:** الاحتمالات الرقمية الخام تُسجل فقط في سجلات `ai_runs` للرقابة الداخلية ولا تُعرض في واجهة المستخدم لتجنب تضليل الفرق.
5. **سياسة التركيز على أعلى $k$ مهام:** ترتيب نسبي داخل السبرنت يركز على أعلى 20% من المهام المعرضة للخطر ($k = \min(5, \lceil 0.2 \times n_{\text{open}} \rceil)$) مع توضيح أهم 3 عوامل مؤثرة.
6. **خط أساس شفاف عند عدم الأهلية:** في حال عدم أهلية السبرنت (أقل من 10 مهام، سبرنت غير نشط، خارج نطاق منتصف السبرنت)، يرجع النظام لقاعدة حتمية واضحة: المهام غير المبدوءة أولاً، ثم الأكثر تعطلاً، ثم الأكبر حجماً.
7. **مهمة فحص يومية مجدولة:** جدولة فحص يومي تلقائي عبر APScheduler مع منع تكرار الاقتراحات المفتوحة أو المرفوضة مسبقاً.
8. **مطابقة عددية:** فحص 200 صف وإثبات التطابق مع حزم البنشمارك بهامش خطأ أقل من $10^{-5}$.

---

### B. What is NOT DONE & Future Scope / ما لم يتم إنجازه والمهام المستقبلية

#### English
1. **Not Validated on Taskove Production Sprints:** High base carry-over rate (~74% in TAWOS) reflects open-source dynamics; performance on closed commercial teams is unmeasured.
2. **Real-time Kafka Event Triggering:** Real-time risk re-scoring on task state change events via Kafka is stubbed; currently relies on REST endpoint triggers and daily scheduled jobs.
3. **Custom Team Thresholds:** The top-20% focus rule is hardcoded globally; team-specific risk tolerance settings are not yet configurable.

#### العربية
1. **لم يتم التحقق على سبرنتات Taskove الإنتاجية:** نسبة التعثر المرتفعة (~74% في TAWOS) تعكس طبيعة المصادر المفتوحة؛ ولم يتم قياس دقة النموذج بعد في فرق الشركات التجارية المغلقة.
2. **ربط أحداث Kafka اللحظية:** تحديث المخاطر لحظياً فور تغير حالة المهمة عبر Kafka لا يزال بحاجة للربط مع الوسيط الفعلي؛ ويعتمد حالياً على استدعاءات الـ REST والفحص اليومي.
3. **تخصيص حساسية الفرق:** نسبة الـ 20% ثابتة حالياً على مستوى النظام ولا يمكن لكل فريق تعديل حساسية المخاطر حسب رغبته.

---

## 4. Suggestions Inbox & State Machine / صندوق الاقتراحات الذكية ودورة اتخاذ القرار

### Overview / نظرة عامة
- **Components / المكونات:** `app/routers/suggestions.py`, `app/models/suggestions.py`.
- **Purpose / الغرض:** Human-in-the-loop review interface for AI recommendations (accept, reject, dismiss, comment, apply).

### A. What is DONE / ما تم إنجازه بالكامل

#### English
1. **Strict RBAC Enforcement:** Scoped strictly to Project Managers (`PM`) and Tech Leads (`TL`) on their accessible projects. The `Admin` role is strictly forbidden (`HTTP 403`).
2. **Atomic State Transitions:** Concurrency conflicts resolved via atomic database updates matching `status == "pending"`; concurrent decisions return `HTTP 409 Conflict`.
3. **Mandatory Decision Reason:** Rejecting or dismissing a suggestion requires a valid `decision_reason` with $\ge 5$ non-whitespace characters (`HTTP 422` if violated).
4. **Two-Phase Apply Flow:** `none` $\to$ `requested` $\to$ `applied` / `failed` ensures safe execution of accepted suggestions with idempotency guards.
5. **Deduplication:** Partial unique indexing prevents duplicate active proposals for the same task and sprint.
6. **No Spurious Hours in Proposals:** Fit suggestions propose de-scoping or moving tasks without relying on linear story-point hour conversions.

#### العربية
1. **تطبيق صارم للصلاحيات:** قصر الوصول على مديري المشاريع (`PM`) وقادة الفرق التقنية (`TL`) على مشاريعهم فقط؛ مع حظر دور المشرف (`Admin`) تماماً بـ `HTTP 403`.
2. **تحولات ذرية للحالة ومنع التعارض:** معالجة تضارب القرارات المتزامنة بتحديث ذري يشترط وجود الحالة `pending`؛ وتوليد خطأ `HTTP 409 Conflict` عند التنازع.
3. **إلزامية ذكر سبب القرار:** اشتراط كتابة سبب مقنع لا يقل عن 5 أحرف عند رفض أو استبعاد أي اقتراح (`HTTP 422` عند المخالفة).
4. **دورة تطبيق ثنائية المراحل:** انتقال الحالة من `none` إلى `requested` ثم `applied` أو `failed` لضمان التنفيذ الآمن لمنع تكرار الإجراءات.
5. **منع التكرار (Deduplication):** فهرسة فريدة تمنع توليد اقتراحات مكررة لنفس المهمة في نفس السبرنت.
6. **استبعاد الساعات الوهمية من المقترحات:** مقترحات نقل المهام تعتمد على سعة الفريق دون اختلاق ساعات وهمية مشتقة من نقاط القصة.

---

### B. What is NOT DONE & Future Scope / ما لم يتم إنجازه والمهام المستقبلية

#### English
1. **Direct Backend Core Mutation:** The apply endpoint marks status as `applied` or `failed`, but does not directly execute task state mutations on the core Taskove database (requires core microservice RPC/Kafka).
2. **Bulk Actions:** No endpoint currently exists to accept or reject multiple suggestions in a single batch request.
3. **Undo/Revert Capability:** Once an accepted proposal is marked applied, there is no automated rollback endpoint.

#### العربية
1. **التعديل المباشر على النظام الأساسي:** نقطة نهاية التطبيق تسجل الحالة `applied` أو `failed`، لكنها لا تقوم بتعديل جداول المهام في قاعدة بيانات Taskove الرئيسية مباشرة (تتطلب ربط RPC أو أحداث Kafka مع الخدمة المركزية).
2. **العمليات الجماعية (Bulk Actions):** لا توجد نقطة نهاية لقبول أو رفض اقتراحات متعددة دفعة واحدة.
3. **التراجع عن الإجراء (Undo/Revert):** لا توجد آلية تراجع تلقائية لإلغاء الاقتراحات بعد تطبيقها.

---

## 5. Sprint Capacity & Fit Check / فحص سعة السبرنت وملاءمة العبء

### Overview / نظرة عامة
- **Components / المكونات:** `app/routers/sprints.py`, `app/services/estimation_service.py`.
- **Purpose / الغرض:** Comparing sprint task workload against team capacity and proposing balance adjustments.

### A. What is DONE / ما تم إنجازه بالكامل

#### English
1. **Three Honest Verdicts:** Categorizes sprint fit into `fits`, `tight`, and `over` based on empirical load ranges against developer capacity.
2. **Dynamic Developer Ratio with Bayesian Shrinkage:** Learns past developer completion ratios from historical work and shrinks small sample sizes toward 1.0.
3. **Elimination of Double-Counting:** Historical task medians are normalized before applying developer ratios, preventing mathematical bias.
4. **Skill-Neutral Proposals:** Task move and reassign suggestions do not evaluate, judge, or rank developer competence.

#### العربية
1. **ثلاثة أحكام واقعية للسبرنت:** تصنيف ملاءمة السبرنت إلى `fits` (مناسب)، `tight` (مضغوط)، و `over` (فائض) بمقارنة واقعية لساعات العمل مع السعة.
2. **معامل سرعة ديناميكي مع تقليص بايزي:** احتساب سرعة إنجاز المطورين من واقع المهام السابقة مع تقليص العينات الصغيرة بايزياً نحو 1.0.
3. **منع الازدواجية الرياضية:** تنقية وسيط المهام التاريخية من سرعة المطورين السابقين لمنع مضاعفة المعاملات الحسابية.
4. **مقترحات محايدة:** مقترحات إعادة التوزيع ونقل المهام محايدة للمهارات ولا تتضمن أي تقييم أو مقارنة لسرعة أو كفاءة المطورين.

---

### B. What is NOT DONE & Future Scope / ما لم يتم إنجازه والمهام المستقبلية

#### English
1. **Live Calendar / Holiday Sync:** Uses a fixed or profile-configured sprint capacity (default 60h); does not sync with real Google/Outlook calendars, leaves, or public holidays.
2. **Skill-Based Reassignment:** Proposals currently prioritize underloaded developers purely on capacity without validating required programming languages or technical domain tags.

#### العربية
1. **المزامنة مع التقويم والإجازات:** تعتمد سعة المطور حالياً على قيمة محددة (افتراضياً 60 ساعة) ولا تتزامن مع تقويمات Google أو عطلات المطورين وإجازاتهم الفعلية.
2. **مراعاة التخصص عند إعادة الإسناد:** المقترحات تعيد التوزيع بناءً على السعة المتاحة فقط دون التحقق التلقائي من لغات البرمجة أو تخصص المطور المطلوب للمهمة.

---

## 6. Agile Drafts & Generative LLMs / مسودات التقارير والذكاء التوليدي

### Overview / نظرة عامة
- **Components / المكونات:** `app/routers/drafts.py`, `app/services/llm_provider.py`.
- **Endpoints / نقاط النهاية:** `/standup/draft`, `/sprints/{id}/retro/draft`, `/release-requests/{id}/draft`, `/release-requests/{id}/explain-block`.

### A. What is DONE / ما تم إنجازه بالكامل

#### English
1. **Multi-Provider Architecture:** Pluggable engine supporting deterministic `template`, local `ollama` (`qwen2.5:3b`), and `cloud` LLMs.
2. **Post-Generation Entity Grounding:** Automatically verifies that every mentioned task ID or developer in the generated draft truly exists in the sprint data.
3. **Prompt Injection Safety:** Encloses untrusted user content within `<UNTRUSTED_DATA>` XML tags with strict system prompt boundaries.
4. **JSON Schema Repair Loop:** Automatically retries once with error feedback if the LLM output violates the Pydantic JSON schema.
5. **No Fabricated Telemetry:** Eliminated old hardcoded constants (`tokens_in=120`, `tokens_out=85`, `latency=0ms`); records real measured latency and actual provider token usage (or `null`).
6. **Developer Privacy & Zero-Ranking:** Strip mood and sentiment analysis fields; prompts forbid ranking or scoring developers.

#### العربية
1. **بنية متعددة المزودين:** محرك مرن يدعم القوالب الحتمية `template`، النماذج المحلية عبر `ollama` (`qwen2.5:3b`)، ومزودي السحابة `cloud`.
2. **التحقق من صحة الكيانات (Entity Grounding):** فحص تلقائي يضمن أن كل معرف مهمة أو اسم مطور ورد في المسودة موجود بالفعل في بيانات السبرنت.
3. **الحماية من حقن الأوامر (Prompt Injection):** تغليف بيانات المستخدم غير الموثوقة داخل وسوم `<UNTRUSTED_DATA>` مع توجيهات صارمة لمنع تجاوز التعليمات.
4. **إصلاح أخطاء مخرجات JSON:** إعادة المحاولة تلقائياً لمرة واحدة مع تزويد النموذج بالخطأ في حال عدم توافق مخرجاته مع مخطط Pydantic.
5. **إلغاء السجلات الوهمية:** حذف القيم الوهمية السابقة نهائياً؛ وتسجيل زمن الاستجابة الفعلي بالمللي ثانية وعدد الرموز الحقيقي الصادر من المزود.
6. **خصوصية المطورين ومنع المقارنة:** حذف حقول قياس الحالة النفسية والمزاجية نهائياً؛ ومنع القوالب من تقييم أو ترتيب المطورين.

---

### B. What is NOT DONE & Future Scope / ما لم يتم إنجازه والمهام المستقبلية

#### English
1. **Custom Fine-Tuned LoRA Weights:** The system uses standard base models (e.g. Qwen 2.5) with in-context prompts; no proprietary fine-tuned agile model is loaded.
2. **Streaming Responses (SSE):** Long draft generation responses are returned in a single HTTP response without chunked streaming.
3. **Multilingual Generation:** Prompts are structured primarily for English draft generation.

#### العربية
1. **عدم وجود نموذج مخصص مدرب (LoRA):** النظام يعتمد على النماذج الجاهزة مع هندسة الأوامر، ولا يوجد نموذج مدرب خصيصاً على بيانات الشركة (LoRA).
2. **عدم دعم البث المباشر (Streaming):** المخرجات الطويلة للمسودات تُعاد في استجابة واحدة كاملة ولا تدعم البث المتدفق (Server-Sent Events).
3. **التوليد باللغة العربية:** القوالب الحالية مصممة ومضبوطة بشكل أساسي لتوليد المسودات باللغة الإنجليزية.

---

## 7. Chat Assistant ("Ask the Project") / المساعد الحواري الذكي للمشروع

### Overview / نظرة عامة
- **Components / المكونات:** `app/routers/chat.py`, `app/services/chat_tools.py`.
- **Endpoint / نقطة النهاية:** `POST /chat`

### A. What is DONE / ما تم إنجازه بالكامل

#### English
1. **Read-Only Grounded Tools:** Routes queries through verified tools (`task_snapshots`, `sprint_snapshots`, `standup_blockers`, `release_requests`).
2. **Honest Negative Answering:** If the database does not contain the answer, responds plainly: *"The project data does not contain this information"* rather than hallucinating.
3. **Session TTL Index:** Automatically purges inactive conversation threads after 7 days using MongoDB TTL indexes.
4. **Developer Isolation & Admin Block:** Developers can only query their own chat sessions; Admin is strictly blocked (`HTTP 403`).

#### العربية
1. **أدوات استعلام للقراءة فقط:** توجيه الاستفسارات حصرياً عبر أدوات موثوقة تبحث في المهام، السبرنتات، المعوقات، وطلبات الإطلاق.
2. **الاعتراف الصادق بعدم توفر المعلومة:** في حال عدم توفر المعلومة في قاعدة البيانات، يجيب النظام بوضوح: *"بيانات المشروع لا تحتوي على هذه المعلومة"* دون اختلاق.
3. **الحذف التلقائي للجلسات الخاملة:** تنظيف جلسات المحادثة القديمة تلقائياً بعد مرور 7 أيام عبر فهارس TTL في MongoDB.
4. **عزل المطورين وحظر المشرف:** المطور لا يرى إلا جلساته فقط؛ والمشرف (`Admin`) محظور تماماً من استخدام المحادثة.

---

### B. What is NOT DONE & Future Scope / ما لم يتم إنجازه والمهام المستقبلية

#### English
1. **Vector RAG over Unstructured Text:** Relies on structured MongoDB attribute searches; lacks semantic vector embeddings (Chroma/Qdrant) over meeting transcripts or documentation.
2. **Multi-Turn Context Compression:** Sessions retain recent messages but lack long-term memory summarization across sprints.

#### العربية
1. **البحث الدلالي المتجهي (Vector RAG):** يعتمد على البحث الهيكلي في قواعد البيانات؛ ولا يحتوي بعد على قاعدة بيانات متجهات (Vector DB) للبحث في الملاحظات غير المنظمة.
2. **تلخيص الذاكرة طويلة المدى:** المحادثة تحفظ الرسائل الأخيرة ولكنها لا تقوم بتلخيص الذاكرة التراكمية عبر عدة سبرنتات متعاقبة.

---

## 8. Architecture, Security & Telemetry / البنية التحتية، الأمان وسجلات الأداء

### A. What is DONE / ما تم إنجازه بالكامل

#### English
1. **Hermetic Test Suite:** 60 passing tests achieving 81% total statement coverage without external dependencies (in-memory `mongomock-motor`).
2. **Role-Based Access Control (RBAC):** Complete security matrix tested and enforced across all endpoints.
3. **Production Startup Gate:** Service immediately halts at boot if `ENVIRONMENT=production` and the insecure default JWT secret is detected.
4. **Rate Limiting:** IP and user-based sliding window rate limits on expensive LLM generation endpoints.
5. **Auditable `ai_runs` Collection:** Every AI inference, fallback, latency measurement, and fallback reason is transparently logged.

#### العربية
1. **حزمة اختبارات معزولة وشاملة:** 60 اختباراً برمجياً ناجحاً تحقق تغطية 81% من الكود دون الحاجة لخادم MongoDB خارجي (استخدام `mongomock-motor`).
2. **مصفوفة أمان الصلاحيات (RBAC):** تأمين صارم لجميع نقاط النهاية واختبار كافة سيناريوهات الأدوار المختلفة.
3. **بوابة أمان الإنتاج:** إيقاف تشغيل الخدمة فوراً عند الإقلاع إذا كانت البيئة `production` وكان المفتاح السري لـ JWT هو المفتاح الافتراضي الضعيف.
4. **تقييد معدل الطلبات (Rate Limiting):** حماية نقاط استدعاء نماذج الذكاء التوليدي بمحددات سرعة زمنية لكل مستخدم وعنوان IP.
5. **سجل عمليات شفاف (`ai_runs`):** تسجيل كل استدعاء، قياس زمني، وحالات الرجوع لخط الأساس في مجموعة بيانات موحدة للرقابة والتدقيق.

---

### B. What is NOT DONE & Future Scope / ما لم يتم إنجازه والمهام المستقبلية

#### English
1. **Live Kafka Cluster Integration:** Kafka event consumers are structured and ready in code, but live ingestion from a production Apache Kafka / Confluent cluster is pending environment setup.
2. **Automated ML Drift Monitoring:** Real-time distribution drift tracking (e.g. via Prometheus/Evidently AI) is not yet wired to alert on concept drift.
3. **Database-per-Tenant Multi-Tenancy:** The current implementation scopes by `project_id` within a shared database collection rather than hard physical database separation per tenant.

#### العربية
1. **الربط مع عنقود Kafka الحي:** أكواد استقبال وبث أحداث Kafka جاهزة ومكتوبة، ولكن ربطها الفعلي مع عنقود Kafka حقيقي بانتظار تجهيز خوادم الإنتاج.
2. **مراقبة انحراف النماذج (Model Drift Monitoring):** عدم وجود نظام مراقبة لحظي آلي (مثل Prometheus أو Evidently) للتنبيه عند تغير خصائص المهام عن بيانات التدريب.
3. **العزل الفيزيائي لبيانات العملاء:** العزل الحالي يتم برمجياً عبر `project_id` داخل نفس المجموعات، وليس بفصل قواعد البيانات كلياً لكل شركة.

---

## 9. Consolidated Scorecard / جدول التقييم والجاهزية الشامل

| Component / المكون | Status / الحالة | Done Highlights / ما تم إنجازه | Pending / Not Done / ما لم يُنجز بعد |
| :--- | :---: | :--- | :--- |
| **Model 1: Story Points** | **Production Ready** | TAWOS LightGBM, out-of-project interval ($c=0.1147$), Fibonacci snapping, type mapping, language guard, zero 500 guarantee. | Validation on private Taskove client data; non-English NLP; online retraining. |
| **Model 2: Carry-Over Risk** | **Production Ready** | TAWOS LightGBM, 23-feature contract, Isotonic calibration, Top-$k$ focus ($k=20\%$), daily automated risk scan. | Commercial team validation; live Kafka event trigger; custom team thresholds. |
| **Suggestions Inbox** | **Production Ready** | Atomic state machine, HTTP 409 conflict safety, $\ge 5$ char reason validation, two-phase apply, Admin blocked. | Direct mutation on core backend database; bulk actions; automated undo. |
| **Sprint Fit & Velocity** | **Production Ready** | Three verdicts (`fits`, `tight`, `over`), Bayesian developer ratio shrinkage, skill-neutral proposals. | Live Google/Outlook calendar integration; skill-matrix requirement matching. |
| **Agile Drafts (LLM)** | **Production Ready** | Multi-provider (`template`, `ollama`, `cloud`), entity grounding, prompt injection defenses, honest token telemetry. | Specialized fine-tuned LoRA weights; SSE streaming responses; Arabic generation. |
| **Chat Assistant** | **Production Ready** | Grounded read-only tools, honest negative answering, 7-day TTL session purge, RBAC developer isolation. | Vector database RAG (unstructured text); multi-sprint memory summarization. |
| **Security & RBAC** | **Production Ready** | Admin blocked on all AI endpoints, default secret startup refusal, sliding window rate limits. | Hard physical database multi-tenancy; SSO/SAML integration. |
| **Test Suite** | **Verified (81%)** | 60/60 passing tests, bit-for-bit parity ($< 10^{-6}$), mutation tests, language guard tests. | Live end-to-end integration tests against actual Kafka brokers and MongoDB clusters. |

---

## 10. Data Architecture, Real Samples & Data Gaps / هيكلية البيانات، عينات تفصيلية، وما ينقص البيانات فعلياً

### 10.1 Data Shapes & Schemas Used / شكل البيانات والمخططات المستخدمة

#### A. Model 1 Training Universe (`tawos_csv/issue.csv`)
- **Raw Volume / الحجم الخام:** 500,000+ Jira issues across 44 open-source software projects.
- **Raw Columns / الأعمدة الأصلية (30 حقل):**
  `['ID', 'Jira_ID', 'Issue_Key', 'URL', 'Title', 'Description', 'Description_Text', 'Description_Code', 'Type', 'Priority', 'Status', 'Resolution', 'Creation_Date', 'Estimation_Date', 'Resolution_Date', 'Last_Updated', 'Story_Point', 'Timespent', 'In_Progress_Minutes', 'Total_Effort_Minutes', 'Resolution_Time_Minutes', 'Title_Changed_After_Estimation', 'Description_Changed_After_Estimation', 'Story_Point_Changed_After_Estimation', 'Pull_Request_URL', 'Creator_ID', 'Reporter_ID', 'Assignee_ID', 'Project_ID', 'Sprint_ID']`
- **Cleaning Filters Applied / عوامل التصفية المطبقة:**
  1. Valid Story Points: $0 < \text{Story\_Point} \le 100.0$ (dropped nulls, zeros, and extreme outliers).
  2. Completed & Valid Resolutions: Excluded won't fix, duplicate, invalid, cannot reproduce.
  3. Pre-Estimation Integrity: `Story_Point_Changed_After_Estimation == 0`, `Title_Changed_After_Estimation == 0`, `Description_Changed_After_Estimation == 0` (preventing retrospective label leakage).
  4. Chronological Splitting: Sorted by `Estimation_Date` / `Creation_Date`. 70% Train, 15% Validation, 15% Test.
  5. Product Scope Filtering: Dropped `Type == "Epic"` (Epics represent multi-month initiatives, not sprint backlog items).
- **Final Clean Partitions / أحجام العينات النهائية:**
  - **Train / التدريب:** 29,330 issues
  - **Validation / التحقق:** 5,177 issues
  - **Test / الاختبار:** 6,167 issues
- **Runtime Model Feature Vector / تمثيل الخصائص أثناء التشغيل:**
  - `Title` + `Description` sliced to max 2,000 characters $\to$ TF-IDF Unigram/Bigram matrix ($2,500$ numerical features).
  - Categorical `Type` $\to$ One-Hot Encoded vector across 17 known Jira categories ($17$ numerical features).
  - Total Sparse Input Vector: **2,517 numerical features**.

---

#### B. Model 2 Training Universe (`benchmark_v2/artifacts/model2/`)
- **Dataset / البيانات:** 92,547 completed sprint tasks across open-source Jira boards.
- **Temporal Cutoff / نقطة الاستقطاع الزمني:** Task state evaluated at elapsed sprint fraction $f = (t - \text{Start}) / (\text{End} - \text{Start})$ with $f \in [0.2, 0.8]$.
- **Target Label / المتغير المستهدف:** Binary carry-over ($1 = \text{task not Done at sprint close}$, $0 = \text{task Done}$). Base carry-over rate $\approx 74.3\%$.
- **Feature Vector Shape / شكل متجه الخصائص:** Exactly **23 numerical features** conforming to `FEATURE_CONTRACT.md`:
  - *Sprint-level (4):* $f$, `sprint_open_tasks`, `sprint_open_points`, `fraction_points_done`.
  - *Team/Assignee-level (2):* `assignee_open_points`, `assignee_open_tasks`.
  - *Task-level (17):* `days_in_current_status`, `days_since_last_event`, `story_points`, `has_sp`, `added_after_sprint_start`, `assignee_present`, `links_depends_on`, `status_category_is_todo`, and 9 one-hot types (`type_Bug`, `type_Story`, `type_Improvement`, `type_Task`, `type_New Feature`, `type_Suggestion`, `type_Enhancement Request`, `type_Epic`, `type_other`).

---

#### C. Live Microservice Database Schemas (MongoDB Read Models)
- **`task_snapshots`:** `{task_id, sprint_id, project_id, title, description, task_type, status, story_points, assignee_id, dependencies, stalled_days, created_at, updated_at}`
- **`sprint_snapshots`:** `{sprint_id, project_id, name, status, start_date, end_date}`
- **`time_entries`:** `{entry_id, task_id, user_id, hours, logged_at}`
- **`capacity`:** `{sprint_id, user_id, capacity_hours}`
- **`ai_profiles`:** `{user_id, estimation: {ratio, sample_size}}`
- **`ai_suggestions`:** `{dedupe_key, type, action, status, apply_status, reason, proposal, change_preview, reasoning_steps, data_sources, decision_reason}`
- **`ai_runs` (Telemetry):** `{feature, model, model_version, method, latency_ms, fallback_reason, tokens_in, tokens_out, status, created_at}`

---

### 10.2 Concrete Data Samples / عينات حقيقية من البيانات

#### Sample 1: Real TAWOS Raw Training Row (Model 1) / عينة حقيقية من بيانات التدريب للنموذج الأول
```json
{
  "ID": 65,
  "Issue_Key": "XD-3768",
  "Type": "Bug",
  "Priority": "Major",
  "Story_Point": 1.0,
  "Title": "How do I make a job restartable in spring xd",
  "Description_Text": "The jobs that appear under Executions section of Jobs in spring xd admin ui, has restart button disabled. How do I enable this ?",
  "Status": "Closed",
  "Resolution": "Done",
  "Creation_Date": "2017-07-10 13:41:25",
  "Estimation_Date": "2017-07-10 12:41:25",
  "Project_ID": 1
}
```

#### Sample 2: Model 1 Production Inference Request & Output / عينة من استدعاء ومخرجات تقدير نقاط القصة
**Client Request Input:**
```python
title = "Fix connection pooling timeout under high concurrent checkout load"
description = "HikariCP pool exhausts connections after 30 seconds of peak traffic in payment service."
task_type = "bug"  # Maps canonically to TAWOS 'Bug'
```
**Model Output Payload:**
```json
{
  "story_points": 2.0,
  "raw_story_points": 1.8419,
  "interval": [1.0, 3.5],
  "method": "lightgbm_taskove_v2",
  "model_version": "v2_taskove",
  "confidence_note": "rough guide for new teams; improves as your team's history grows",
  "fallback_reason": null
}
```

#### Sample 3: Real 23-Feature Vector (Model 2 Sprint Carry-Over) / عينة من متجه الخصائص الـ 23 للنموذج الثاني
Task mid-sprint ($f = 0.50$, Todo status, 3 Story Points, assigned, 2 blockers):
```python
X = [
    0.50,  # 1. f (50% of sprint elapsed)
    14.0,  # 2. sprint_open_tasks
    38.0,  # 3. sprint_open_points
    0.35,  # 4. fraction_points_done (35% done)
    8.0,   # 5. assignee_open_points
    3.0,   # 6. assignee_open_tasks
    4.2,   # 7. days_in_current_status
    1.1,   # 8. days_since_last_event
    3.0,   # 9. story_points
    1.0,   # 10. has_sp (True)
    0.0,   # 11. added_after_sprint_start (committed at planning)
    1.0,   # 12. assignee_present (assigned)
    2.0,   # 13. links_depends_on (2 predecessor blockers)
    1.0,   # 14. status_category_is_todo (still in Todo)
    0.0,   # 15. type_Bug
    1.0,   # 16. type_Story (Feature)
    0.0,   # 17. type_Improvement
    0.0,   # 18. type_Task
    0.0,   # 19. type_New Feature
    0.0,   # 20. type_Suggestion
    0.0,   # 21. type_Enhancement Request
    0.0,   # 22. type_Epic
    0.0    # 23. type_other
]
# Output Calibrated Carry-Over Probability: 0.7821 -> Ranked into High Risk Tier (Top-k Focus)
```

#### Sample 4: Sprint Estimation Endpoint Response (`POST /sprints/{id}/estimate`)
```json
{
  "sprint_id": "sprint_01",
  "tasks_estimated": 3,
  "total_low_hours": 12.5,
  "total_mid_hours": 18.0,
  "total_high_hours": 24.5,
  "estimates": [
    {
      "task_id": "task_101",
      "story_points": 3.0,
      "story_points_interval": [2.0, 5.0],
      "story_points_confidence_note": "rough guide for new teams; improves as your team's history grows",
      "sp_method": "lightgbm_taskove_v2",
      "sp_model_version": "v2_taskove",
      "low_hours": 4.5,
      "mid_hours": 6.0,
      "high_hours": 8.5,
      "hours_conversion_note": "unvalidated heuristic (weak SP-to-time correlation r~0.06-0.30; hours conversion not empirical)",
      "confidence": "high",
      "method": "rule_baseline",
      "model_version": "deterministic_baseline_v1"
    }
  ]
}
```

---

### 10.3 What is ACTUALLY Missing / ما ينقص البيانات فعلياً

#### 1. Zero Production Data from Taskove Customers / غياب بيانات عملاء Taskove الفعليين كلياً
* **The Reality / الواقع:**
  * **100% of the training data** comes from public open-source Jira repositories (Apache Software Foundation, Spring Framework, Mozilla, JBoss) from the years **2005 to 2018**.
  * The service has **never** processed or trained on a single real task, sprint, or story-point from a paying, commercial Taskove tenant.
  * *الفجوة الحقيقية:* جميع النماذج تم تدريبها بنسبة 100% على مشاريع مفتوحة المصدر قديمة (2005-2018)، ولم يرَ النموذج أي مهمة أو سبرنت حقيقي لعملاء Taskove الفعليين.

#### 2. Open-Source vs. Commercial Agile Dynamics / الفجوة الثقافية بين المصادر المفتوحة وفرق الشركات
* **The Reality / الواقع:**
  * In TAWOS, the baseline sprint carry-over rate is **74.3%** because contributors are global volunteers working asynchronously without contractual sprint commitments; tasks routinely carry over across months or years.
  * In commercial software companies using Taskove, a sprint carry-over rate above 20–30% triggers team retrospective alarms.
  * *الفجوة الحقيقية:* نسبة تعثر المهام في بيانات التدريب تفوق 74% نظراً لطبيعة المتطوعين، بينما في الشركات التجارية لا تتعدى النسبة 20-30%؛ مما يجعل النموذج يميل للحذر المفرط.

#### 3. Vocabulary & Tech Stack Drift / انحراف المعجم والمصطلحات البرمجية
* **The Reality / الواقع:**
  * TAWOS vocabulary is heavily dominated by mid-2010s Java backend infrastructure (Maven, Gradle, JVM, classpath, Zookeeper, OSGi, Tomcat, Hadoop).
  * Modern Taskove teams write tickets about React Hooks, Next.js App Router, Tailwind CSS, Docker Compose, Supabase, Tailwind, Flutter, or GraphQL. These modern tokens are poorly represented or completely out-of-vocabulary in the TF-IDF feature space.
  * *الفجوة الحقيقية:* كلمات معجم الـ TF-IDF تنتمي لتقنيات جافا القديمة، وتفتقر للمصطلحات البرمجية الحديثة لمطوري الويب والموبايل اليوم.

#### 4. Multilingual Deficit / غياب النصوص غير الإنجليزية كلياً
* **The Reality / الواقع:**
  * TAWOS contains **0% non-English issues**.
  * Any user in Taskove writing ticket titles or descriptions in Arabic, German, French, or Japanese cannot leverage the ML text weights and is diverted by the Language Guard to static median fallback tables.
  * *الفجوة الحقيقية:* انعدام تام لأي بيانات تدريب بلغات أخرى (مثل العربية)، مما يمنع النماذج من استيعاب معاني المهام المكتوبة بلغة غير الإنجليزية.

#### 5. Deleted Jira Fields Missing in Taskove / حقول كانت موجودة في TAWOS وحُذفت لعدم وجودها في Taskove
* **`Priority`:** In Jira, Priority was one of the strongest predictive features (explaining ~0.15 of MAE). It was stripped because Taskove's task model does not contain Priority.
* **`Components` & `Fix_Versions`:** Project subsystem tags present in Jira were discarded.
* **Detailed Changelog Table:** TAWOS had every timestamped transition between statuses. Taskove only maintains `stalled_days` and an `updated_at` timestamp.
* **`sprint_assigned_at`:** Taskove does not record the exact timestamp when a backlog item was assigned to an active sprint, forcing the model to approximate with `created_at`.
* *الفجوة الحقيقية:* التخلي عن متغيرات مؤثرة جداً (مثل درجة الأولوية، سجل التحولات الزمني الدقيق، وتاريخ إضافة المهمة للسبرنت) لعدم وجودها في هيكل بيانات Taskove.

#### 6. Missing Real-World Context in TAWOS / بيانات يحتاجها النظام ومفقودة كلياً في بيانات التدريب
* **Real Developer Calendars & PTO:** Neither TAWOS nor the current service knows if a developer has 2 days of sickness, public holidays, or vacations during a sprint. Capacity is naively assumed as 60 hours.
* **Repository & Codebase Metadata:** No link between tasks and actual pull requests, lines of code changed, or number of files touched.
* **Absence of a Human Feedback Loop:** When Model 1 predicts 5 SP and the team changes it to 3 SP during planning, the service currently logs telemetry but does **not** persist that delta into an active retraining loop.
* *الفجوة الحقيقية:* غياب الربط مع إجازات المطورين الفعلية، غياب الربط مع مستودعات Git (أكواد الـ PR)، وغياب حلقة التعلم التفاعلي المستمر (Active Feedback Loop) من تعديلات المستخدمين.


