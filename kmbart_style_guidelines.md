# KM-BART Academic Writing Style Guidelines

**Purpose**: This document provides a comprehensive template for transforming AI-generated academic text into authentic, human-written scholarly work that matches the style of the KM-BART thesis chapter. The target voice is a master's student whose second language is English and first language is Chinese.

**Document Version**: 1.0
**Created**: 2025-10-02
**Based on**: kmbart.tex language analysis + humanization protocol

---

## TABLE OF CONTENTS
1. [TARGET STYLE CHARACTERISTICS](#part-i-target-style-characteristics)
2. [CORE HUMANIZATION PRINCIPLE](#part-ii-core-humanization-principle)
3. [DIAGNOSTIC AUDIT PROTOCOL](#part-iii-diagnostic-audit-protocol)
4. [TRANSFORMATION PROTOCOLS](#part-iv-transformation-protocols)
5. [OPERATIONAL GUARDRAILS](#part-v-operational-guardrails)
6. [PASS INTENSITY PRESETS](#part-vi-pass-intensity-presets)
7. [OUTPUT TEMPLATE](#part-vii-output-template)
8. [PRACTICAL WORKFLOW](#part-viii-practical-workflow)
9. [APPENDICES](#appendix-a-banned-words--replacements)

---

## PART I: TARGET STYLE CHARACTERISTICS (kmbart.tex Analysis)

### 1. Sentence Structure & Length
**Characteristics**:
- **Consistent medium-to-long sentences** (15-30 words typical)
- **Complex, multi-clause constructions** with subordinate clauses
- **No sentence fragments** - all sentences are grammatically complete
- **Smooth, flowing transitions** between ideas

**Example**:
```
"We adapt the generative BART architecture~\cite{bart} to a multimodal
model with visual and textual inputs."
```

### 2. Tone & Voice
**Characteristics**:
- **Formal academic tone** throughout
- **First-person plural ("We")** as primary subject
- **Confident, assertive** statements
- **Professional and polished** - no casual language

**Example**:
```
"To the best of our knowledge, we are the first to propose a dedicated
task for improving model performance on the VCG task."
```

### 3. Vocabulary & Terminology
**Characteristics**:
- **Technical precision**: Transformer-based, sequence-to-sequence, multimodal, cross-modal encoder
- **Domain-specific jargon** used naturally
- **Abstract nouns**: alignment, semantics, reasoning, architecture
- **Formal verbs**: leverage (technical), acquire, enable, demonstrate, conduct
- **No colloquialisms** or informal expressions

**Preferred Academic Verbs**:
- propose, introduce, present, describe
- demonstrate, show, achieve, reach
- improve, enhance, boost (with numbers)
- employ, use, adopt, leverage (technical)
- conduct, perform, execute
- observe, find, note

### 4. Transitions & Flow
**Characteristics**:
- **Explicit transition words**: However, Therefore, Furthermore, For instance, To ease this problem
- **Smooth logical progression**: ideas build on each other systematically
- **Clear signposting**: "In this section," "The following subsections," "We describe..."
- **Cohesive devices**: "These models," "Such information," "This indicates"

**Standard Transitions**:
- **Sequential**: Furthermore, Moreover, Additionally, In addition
- **Contrastive**: However, Nevertheless, In contrast, On the other hand
- **Causal**: Therefore, Consequently, Thus, As a result
- **Exemplification**: For instance, For example, Specifically

### 5. Paragraph Organization
**Characteristics**:
- **3-5 sentences per paragraph** typically
- **Topic sentence + supporting details** structure
- **No abrupt breaks** or fragmentary paragraphs
- **Logical development** of each idea before moving to the next

**Standard Pattern**:
1. Topic sentence (main claim)
2. Supporting evidence/details (2-3 sentences)
3. Explanation/analysis (1-2 sentences)
4. Transition to next idea (optional)

### 6. Rhetorical Features
**Characteristics**:
- **No rhetorical questions** (or extremely rare)
- **No exclamations** or emphatic punctuation
- **Enumerated lists** for clarity (a., b., c.)
- **Parallel structure** in enumerations

**Example**:
```latex
\begin{itemize}
    \item[a.] We extend the BART model to process multimodal data...
    \item[b.] To improve the model performance on VCG, we implicitly
              incorporate commonsense knowledge...
    \item[c.] Besides KCG, we further equip our KM-BART with standard
              pretraining tasks...
\end{itemize}
```

### 7. Sentence Rhythm
**Characteristics**:
- **Predictable, steady rhythm** - no dramatic variations
- **Consistent pacing** throughout sections
- **No staccato effects** or very short sentences for emphasis
- **Academic cadence** - measured and methodical

### 8. Citation Integration
**Characteristics**:
- **Citations embedded smoothly**: `~\cite{bart}` or `~\parencite{vqa}`
- **Multiple citations grouped**: `~\cite{lxmert, vilbert, uniter, ernie-vil}`
- **Natural flow** - citations don't interrupt reading

**Citation Patterns**:
- After claims: "...reaches state-of-the-art performance~\cite{vcg}."
- After methods: "Following previous work~\cite{lxmert, vilbert}, we use..."
- After definitions: "ConceptNet~\cite{concept} is a knowledge graph..."

### 9. Language Characteristics
**Characteristics**:
- **Active voice predominant** with "We" as agent
- **Passive voice** used appropriately for processes/methods
- **Complex noun phrases**: "the necessary level of practical knowledge and reasoning"
- **Conditional/causal relationships**: "However, only feature alignments are inadequate..."

**Voice Distribution**:
- Active (70%): "We propose," "We use," "We observe"
- Passive (30%): "The model is pretrained on," "Results are shown in"

### 10. Precision & Polish
**Characteristics**:
- **Highly polished** - appears edited and refined
- **No contractions** (it's → it is)
- **Precise technical terms** consistently used
- **Minimal hedging** beyond standard academic qualifiers
- **Clean presentation** - no informal asides

### 11. Structural Patterns
**Characteristics**:
- **Background → Problem → Solution** narrative arc
- **Systematic methodology** descriptions
- **Detailed experimental** reporting with tables/figures
- **Formal conclusions** that summarize contributions

---

## PART II: CORE HUMANIZATION PRINCIPLE

**Authenticity over Perfection**

Your goal is a credible, human-sounding thesis, not a flawlessly optimized one. Human writing contains minor imperfections, idiosyncrasies, and a personal rhythm. You must introduce these elements deliberately.

**Key Insight**: The kmbart.tex style represents *polished* academic writing, but even polished human writing has subtle characteristics that distinguish it from AI generation. The goal is to match this professional academic style while avoiding AI detection patterns.

**Important**: For highly formal sections (like kmbart.tex), use Low or Medium Pass. For less formal sections, Strong/Extreme/Ultra Pass may be more appropriate.

---

## PART III: DIAGNOSTIC AUDIT PROTOCOL

**Before any editing**, analyze the source text and create a brief report identifying these "AI Tells":

### 1. Sentence Rhythm
- **Check**: Is there a monotonous, predictable pattern?
- **AI Pattern**: Uniform 20-25 word sentences throughout (85%+ uniformity)
- **Human Pattern**: Variation within consistency (15-30 words, occasional shorter/longer)

### 2. Lexical Choice
- **Check**: Over-reliance on certain "AI-preferred" words
- **Banned Words List**:
  - **Hype words**: tapestry, delve, paradigm, realm, landscape, holistic, groundbreaking, cutting-edge
  - **Overused**: robust/robustness (as hype), leverage (as hype), foster, underscore(s), myriad, plethora
  - **Vague**: empower, pivotal, endeavor, comprehensive (as fluff), novel (as hype)

### 3. Hedging Analysis
- **Check**: Blanket use of "may," "could," "might," "it is suggested that"
- **AI Pattern**: Generic hedging without specificity
- **Human Pattern**: Precise qualifiers tied to evidence ("in our experiments," "for the 0.6B model," "with our setup")

### 4. Structural Predictability
- **Check**: Rigid, formulaic paragraph structures
- **AI Pattern**: Every paragraph = Topic → Evidence → Explanation → Link
- **Human Pattern**: Varied entry points (sometimes start with evidence, sometimes with result, sometimes with contrast)

### 5. Excessive Flow
- **Check**: Lack of friction, abrupt transitions, or occasional digressions
- **AI Pattern**: Perfectly smooth, optimized flow throughout
- **Human Pattern**: Occasional natural friction, varied connectors, minor digressions

### 6. Syntactic Perfection
- **Check**: Absence of stylistic variation
- **AI Pattern**: All sentences follow strict grammatical rules
- **Human Pattern**: Occasional intentional variations (rare in kmbart.tex formal style)

---

## PART IV: TRANSFORMATION PROTOCOLS

### Protocol 1: Stylistic Rewriting Engine

#### A. Vary Sentence Architecture
**Target**: Mix sentence lengths while maintaining academic formality

**kmbart.tex standard**: 15-30 words, mostly complex sentences

**Action**:
- Keep 70% in the 15-30 word range
- Allow 20% slightly longer (30-35 words)
- Allow 10% shorter (10-15 words) for emphasis

**Example Transformation**:
```
AI: "The model demonstrates significant improvements across multiple
metrics, indicating that the proposed approach may be effective for
this task."

kmbart.tex style: "Our model achieves significant improvements across
BLEU-2, METEOR, and CIDER metrics (Table 4). This demonstrates the
effectiveness of our proposed pretraining tasks."
```

#### B. Purge & Replace AI Lexicon
**Target**: Remove AI-preferred words; use field-appropriate terminology

**Replacement Guide**:
| AI Word | kmbart.tex Alternative |
|---------|------------------------|
| robust (as hype) | stable, consistent, reliable, effective |
| leverage (as hype) | use, apply, utilize, employ |
| paradigm | approach, model, framework, method |
| foster | enable, support, facilitate, promote |
| tapestry | combination, integration, structure |
| realm/landscape | domain, area, field |
| holistic | comprehensive (only if specific) |
| novel (as hype) | new (factual), proposed |
| underscore | demonstrate, show, indicate, emphasize |
| myriad/plethora | many, multiple, various, numerous |
| groundbreaking | significant, substantial (with evidence) |
| cutting-edge | recent, current, advanced (if factual) |
| empower | enable, allow, support |
| pivotal | key, important, critical, essential |
| endeavor | effort, work, attempt, study |

#### C. Word Style for L2 English (Chinese L1)
**Target**: Natural but slightly formal vocabulary choices

**Characteristics** (subtle, use sparingly):
- Prefer straightforward verbs: "conduct experiments" vs. "carry out experiments"
- Use formal connectors: "therefore" preferred over "so"
- Explicit transitions over implicit ones
- Direct constructions: "We use X to achieve Y"

**Note**: kmbart.tex is already very polished, so L2 markers should be minimal.

#### D. Refine Hedging
**Target**: Precise, confident language with evidence-based qualifiers

**kmbart.tex pattern**: Minimal hedging; confident assertions backed by results

**Transformation Examples**:
```
AI Generic: "The results may suggest a correlation between X and Y."
kmbart.tex: "The results demonstrate a correlation between X and Y
(see Table 3)."

AI Vague: "This could potentially improve performance."
kmbart.tex: "This improves performance by 15% on the VCG validation
set (Table 4)."

AI Overhedged: "It might be possible that the model learns better
representations."
kmbart.tex: "The model learns better multimodal representations, as
evidenced by improved CIDER scores (39.76 vs. 39.13)."
```

**Acceptable Hedges** (when justified):
- "This suggests..." (when evidence is preliminary)
- "We argue that..." (for interpretive claims)
- "This indicates..." (for inferential results)
- "This demonstrates..." (when results are clear)

#### E. Embrace Controlled "Imperfections"
**Target**: Rare, intentional stylistic choices that add human rhythm

**kmbart.tex standard**: Minimal imperfections; highly polished

**Acceptable Variations** (rare, 1-2 per section maximum):
- "And" sentence opener for continuation (very rare)
- Slightly unwieldy long sentence (occasional)
- Natural transitions: "To ease this problem," "Put another way"

**Important**: kmbart.tex is very formal. Use these sparingly to avoid over-humanizing.

---

### Protocol 2: Structural Re-engineering

#### A. Break Formulaic Paragraphs
**Target**: Varied paragraph structures while maintaining logic

**kmbart.tex pattern**: Primarily Topic → Support, but with natural variations

**Variation Strategies**:
1. **Standard**: Topic sentence → Evidence → Explanation
2. **Result-first**: Finding → Explanation → Implication
3. **Problem-solution**: Issue → Our approach → Outcome
4. **Contrast**: Previous work → Gap → Our contribution

**Example from kmbart.tex**:
```
Result-first approach:
"KM-BART achieves state-of-the-art performance on the VCG task. The
model is based on a Transformer encoder-decoder architecture,
pretrained on novel tasks including Knowledge-based Commonsense
Generation (KCG)."
```

#### B. Introduce "Personal Voice" Layer
**Target**: Signal researcher presence without excessive first-person

**kmbart.tex pattern**: "We" as primary agent; active voice; decision markers

**Personal Voice Markers**:
- "We propose..." / "We introduce..." / "We present..."
- "To the best of our knowledge, we are the first..."
- "We observe that..." / "We find that..." / "We demonstrate..."
- "We employ..." / "We use..." / "We leverage..."

**Decision Signaling**:
```
"To ease this problem, we use..."
"Following previous work~\cite{X}, we adopt..."
"We further equip our model with..."
"We argue that such information is not generally available..."
```

#### C. Integrate "Research Narrative"
**Target**: Brief, justifiable explanations for methodological choices

**kmbart.tex examples**:
```
"We only use COMET to generate new commonsense descriptions on SBU
and COCO datasets due to limits in computational power for
pretraining."

"We argue that such information [Place and Person] is not generally
available in normal settings, where only images and event
descriptions are given. Hence, we do not use the Place and Person
information in our KM-BART."
```

**Pattern**: Decision → Justification → Impact

---

### Protocol 3: Factual Grounding & Citation Integrity

#### A. Flag Hallucinations
**Action**: Identify unsupported claims, fabricated citations, inferred data

**Tag Format**: `[REQUIRES VERIFICATION: specific claim or citation]`

**Common Hallucination Types**:
1. Fabricated citation keys
2. Invented statistics without source
3. Inferred relationships not present in data
4. Generalized claims without evidence

#### B. Demand Specificity
**kmbart.tex standard**: Exact details, precise numbers, explicit methods

**Required Specifics**:
- **Sample sizes**: "n = 30 examples," "146K validation examples"
- **Hyperparameters**: "learning rate of 1e-5," "dropout rate of 0.1"
- **Results**: "CIDER score of 39.76," "BLEU-2 of 23.47"
- **Training**: "20 epochs," "batch size of 256," "4 Titan RTX GPUs"
- **Effect sizes**: "improves performance by 15%," "66.7% of cases"
- **Statistical significance**: p-values, confidence intervals when appropriate

**Example**:
```
AI Vague: "We trained the model on a large dataset for many epochs."

kmbart.tex: "We pretrain our model for 20 epochs on 4 Titan RTX GPUs
with an effective batch size of 256."
```

#### C. Annotate Sources
**Action**: Integrate citations naturally; add brief context when helpful

**kmbart.tex patterns**:
```
Standard: "Following previous work~\cite{lxmert, vilbert}, we use a
convolution neural network pretrained on the COCO dataset to extract
visual embeddings."

With context: "We leverage COMET~\cite{comet}, which is a
Transformer-based, generative model pretrained on commonsense
knowledge graphs including ConceptNet and Atomic."
```

---

## PART V: OPERATIONAL GUARDRAILS

### 1. NO New Content Generation
- **Rule**: You are an editor, not a co-author
- **Action**: Reframe existing content; do not invent theories, analyses, or conclusions
- **Exception**: Minor clarifications that make existing content clearer (without changing meaning)

### 2. Preserve Core Argument
- **Rule**: Original thesis statement, findings, and intellectual contribution are sacrosanct
- **Action**: Reclothe them in human voice; maintain logical structure
- **Preserve**: All numerical results, experimental conditions, factual claims, citations

### 3. Maintain LaTeX Integrity
- **Rule**: Do not modify structural LaTeX elements
- **Preserve**:
  - Section/subsection labels: `\section{}`, `\subsection{}`
  - Cross-references: `\Cref{}`, `\ref{}`, `\eqref{}`
  - Citation keys: `\cite{}`, `\parencite{}`, `\textcite{}`
  - Math environments: equations, `\mathbb{}`, `\theta`, etc.
  - Figure/table references: `Figure~\ref{fig:X}`, `Table~\ref{tab:Y}`
  - Custom macros and commands
  - Labels for figures, tables, equations

### 4. Transparency Log
- **Action**: Keep a running list of all major changes
- **Format**: Pattern Found → Action Taken

**Example Log Entry**:
```
Pattern: Repetitive "Furthermore," sentence opener (8 occurrences)
Action: Replaced with varied transitions: "Additionally," "Moreover,"
        "We also observe," "In addition"
```

---

## PART VI: PASS INTENSITY PRESETS

Choose based on the formality level of the source text and desired humanization strength.

### Low Pass (Subtle Polish)
**Purpose**: Light de-AI for near-final text; preserve formal tone and structure

**Best for**: Already polished text, formal academic sections, methodology

**Scope**:
- **Sentence rhythm**: Slight variation; break only longest chains; no fragments
- **Lexicon**: Replace obvious AI words; keep field terms; minimal synonym changes
- **Hedging**: Calibrate extremes (replace generic "may" with "indicates" when justified); keep conservative tone
- **Transitions**: Keep formal connectors; add 1-2 conversational pivots maximum
- **Structure**: No paragraph reordering; minimal topic sentence adjustments
- **Personal voice**: Light first-person ("we") where already implied; avoid strong authorial asides

**Deliverables**:
1. Brief Diagnostic Summary (3-4 bullet points)
2. Concise Revision Log
3. Short Verification Checklist

---

### Medium Pass (Noticeable but Restrained)
**Purpose**: Clear human cadence with mild friction; moderate cleanup; minor structure edits

**Best for**: Good AI text that needs moderate humanization; results sections

**Scope**:
- **Sentence rhythm**: Mix long/short; allow 1-2 "And/But" openers; 1-2 rhetorical fragments where helpful
- **Lexicon**: Trim grandiose/AI-ish terms; prefer plain, field-appropriate words; keep precision
- **Hedging**: Replace vague hedges with specific qualifiers ("in our setup," "preliminary," "with this dataset")
- **Transitions**: Use a few conversational pivots ("Put another way," "In practice," "Still,") to break monotony
- **Structure**: Reorder within paragraphs if clarity improves; keep section order; occasionally start with result first
- **Personal voice**: Add researcher presence (choices, surprises, limitations) without new claims

**Deliverables**:
1. Full Diagnostic Summary
2. Targeted Revision Log (key changes with examples)
3. Explicit Verification Checklist

---

### Strong Pass (Heavy De-AI; Preserve Meaning)
**Purpose**: Maximal removal of AI fingerprints while keeping all facts, claims, conclusions intact

**Best for**: AI text with heavy AI patterns; discussion sections

**Scope**:
- **Sentence rhythm**: Aggressively split long sentences; frequent short clauses; allow several "And/But" openers; selective fragments
- **Lexicon**: Broad cleanup of AI-preferred words; simpler connectors; favor direct, concrete phrasing over abstractions
- **Hedging**: Tighten to specific, evidence-bound qualifiers; avoid generic vagueness; keep honest limits
- **Transitions**: Add mild friction and natural asides ("Not always." "A simple reason." "Still,") sparingly
- **Structure**: Break formulaic patterns; reorder paragraph internals; occasionally lead with evidence or finding; keep argument intact
- **Personal voice**: Clear first-person researcher presence (decisions, surprises, caveats) without inventing content

**LaTeX/Citations**:
- Remove all inline `[REQUIRES VERIFICATION]` notes from chapter text
- Track every item only in external Verification Checklist

**Deliverables**:
1. Diagnostic Summary
2. Detailed Revision Log (what changed and why)
3. Standalone Verification Checklist

---

### Extreme Pass (Maximal Humanization; Meaning Unchanged)
**Purpose**: Strongest stylistic and structural humanization without altering claims, numbers, or citations

**Best for**: Heavily AI-generated text needing maximum de-AI; introduction/conclusion sections

**Scope**:
- **Sentence rhythm**: Very high variation. Split most long sentences. Frequent short sentences. Several "And/But" openers. Intentional rhetorical fragments where emphasis helps.
- **Lexicon**: Extensive removal of AI-preferred phrasing. Prefer concrete, plain verbs and tight nouns. Keep technical terms; reduce nominalizations and abstractions.
- **Hedging**: Anchor to evidence and scope ("in this setup," "with our data," "for 0.6B–4B"). Eliminate generic hedges not tied to results.
- **Transitions**: Introduce natural friction and reflective asides ("Still," "To be fair," "In practice,") used judiciously
- **Structure**: Bold reorganization inside paragraphs. Lead with results or evidence when clearer. Interleave limitations earlier where appropriate. Do NOT change section/subsection order, labels, or LaTeX cross-refs.
- **Personal voice**: Distinct researcher presence (decisions, surprises, constraints). No new content, no new claims.

**Formatting/LaTeX**:
- Do NOT modify: equations, labels, figure/table numbers, citation keys, macros
- Keep all `\Cref` references intact
- Maintain all mathematical notation

**Citations & Verification**:
- No inline `[REQUIRES VERIFICATION]` notes
- Maintain separate Verification Checklist outside the chapter
- If statement appears unsupported, move to checklist for user to confirm post-edit

**Deliverables**:
1. Full Diagnostic Summary
2. Granular Revision Log (per paragraph where meaningful)
3. Refreshed Banned Words List
4. Complete Verification Checklist

---

### Ultra Pass (Beyond Extreme)
**Purpose**: Maximal humanization and detection resistance while preserving every claim, number, equation, label, reference, citation

**Best for**: Thesis chapters that need to pass strict AI detection while maintaining full academic integrity

**Additional Scope Beyond Extreme**:
- **L2-English (Chinese L1) cadence**: Slightly simpler phrasing and direct verbs; occasional article/preposition stiffness (sparingly) while maintaining academic clarity. Prefer straightforward connectors ("So," "But," "Still,")
- **Banned words (expanded)**: All Extreme list + state-of-the-art (as hype), groundbreaking, cutting-edge, empower, pivotal, endeavor
- **Preferred alternatives**: consistent/stable, use/apply, approach/model, support/enable, domain/area, overview, new (factual), show/indicate, many, allow, key, major, thorough (when evidenced)
- **Sentence rhythm**: Very high variation; frequent short mixed with longer, slightly unwieldy ones; several "And/But/So/Still" openers; occasional fragments; 1-2 reflective asides per section
- **Structure**: Bold reorganization within paragraphs; vary entry points (lead with result or evidence); alternate evidence→claim and claim→evidence flows

**Scope & Limits**:
- No new content, numbers, or citations
- Do NOT alter: section/subsection order, labels, macros, equations, or filenames
- Rework only within paragraphs and sentences

**Deliverables**:
1. Per-chapter Diagnostic Summary
2. Detailed Revision Log (paragraph-level where meaningful)
3. Updated Banned Words List
4. Standalone Verification Checklist
5. **Optional**: One-page "Style Rationale" explaining transformation approach

---

## PART VII: OUTPUT TEMPLATE

For each chapter/section processed, return the following:

### 1. Diagnostic Summary
A 3-4 bullet point list of primary "AI Tells" identified

**Template**:
```
**Diagnostic Summary: [Chapter/Section Name]**

- Sentence Rhythm: [Pattern observed, e.g., "Monotonous 20-25 word
  sentences throughout (85% uniformity)"]
- Lexical Choice: [AI words found, e.g., "Overuse of 'robust' (7×),
  'leverage' (5×), 'paradigm' (3×)"]
- Hedging: [Pattern, e.g., "Generic 'may/could' hedging without
  specificity (12 instances)"]
- Structure: [Pattern, e.g., "Rigid Topic→Evidence→Explanation pattern
  in all paragraphs"]
```

### 2. Revision Log
Concise table or list summarizing changes

**Format**:
| AI Pattern Found | Humanizing Action Taken |
|------------------|-------------------------|
| Repetitive "Furthermore," opener (8×) | Replaced with: "Additionally," "Moreover," "We also observe," "In addition" |
| Vague hedge: "may suggest" (6×) | Replaced with: "indicates," "demonstrates" (with evidence references) |
| AI word: "robust" (7×) | Replaced with: "stable," "consistent," "reliable," "effective" |
| Uniform sentence length (20-25 words, 85%) | Varied to: 15-30 words (70%), shorter (15%), longer (15%) |

**Alternative List Format**:
```
**Revision Log**
1. Sentence Rhythm: Split 12 long sentences; added 3 short emphatic
   sentences
2. Lexicon: Replaced "robust"→"consistent" (5×), "leverage"→"use" (4×),
   "paradigm"→"approach" (3×)
3. Hedging: Replaced generic hedges with specific qualifiers tied to
   experimental scope
4. Structure: Reordered 6 paragraphs to lead with findings instead of
   topic sentences
```

### 3. Verification Checklist
List of all `[REQUIRES VERIFICATION]` flags and factual details needing confirmation

**Template**:
```
**Verification Checklist**

1. [REQUIRES VERIFICATION: Citation "Smith et al. 2020" - confirm
   this source exists and supports claim about X]
2. [REQUIRES VERIFICATION: "15% improvement" - confirm number against
   actual experimental results in Table 4]
3. Hyperparameter confirmation: Is learning rate 1e-5 or 2e-5? (text
   says 1e-5, verify against training logs)
4. Sample size: Text says "n=30" - confirm this matches actual
   experimental setup
5. Statistical significance: Add p-value for "significant improvement"
   claim in line 145
```

---

## PART VIII: PRACTICAL WORKFLOW

### Step-by-Step Process

#### Step 1: Diagnostic Phase
```
a. Read the full AI-generated chapter/section
b. Identify AI patterns using PART III checklist
c. Note specific instances (sentence examples, word counts)
d. Create Diagnostic Summary
```

**Checklist**:
- [ ] Sentence rhythm analyzed (check for uniformity)
- [ ] Lexical choice reviewed (count AI-preferred words)
- [ ] Hedging patterns identified (generic vs. specific)
- [ ] Structure examined (formulaic vs. varied)
- [ ] Flow assessed (overly smooth vs. natural)
- [ ] Syntactic patterns noted

#### Step 2: Calibration Phase
```
a. Select appropriate Pass Intensity based on:
   - Formality level required (kmbart.tex = Low/Medium)
   - Amount of AI fingerprinting detected
   - Section type (Methods = Low, Discussion = Strong)
b. Review kmbart.tex style targets for this section type
c. Identify priority patterns to address
d. Plan transformation approach
```

**Selection Guide**:
- **Low Pass**: Methodology, formal results, already polished text
- **Medium Pass**: Most chapters, moderate AI patterns
- **Strong Pass**: Introduction, discussion, heavy AI patterns
- **Extreme Pass**: Sections needing maximum de-AI
- **Ultra Pass**: Entire thesis requiring detection resistance

#### Step 3: Transformation Phase
```
a. Apply Protocol 1: Stylistic Rewriting
   - Vary sentence lengths (target distribution)
   - Replace AI lexicon (use replacement guide)
   - Refine hedging (tie to evidence)

b. Apply Protocol 2: Structural Re-engineering
   - Vary paragraph entry points
   - Add researcher voice markers
   - Integrate research narrative

c. Apply Protocol 3: Factual Grounding
   - Verify all numbers, citations
   - Add specificity (sample sizes, hyperparameters)
   - Flag uncertain claims for verification
```

#### Step 4: Quality Check Phase
```
a. Compare against kmbart.tex style targets
b. Verify all LaTeX elements intact:
   - Labels (\section, \subsection, \label)
   - Cross-references (\Cref, \ref)
   - Citations (\cite, \parencite)
   - Math (equations, symbols)
c. Confirm no new content added
d. Check that core argument preserved
e. Verify all numbers and claims unchanged
```

**LaTeX Check**:
- [ ] All section labels preserved
- [ ] All cross-references working
- [ ] All citation keys unchanged
- [ ] All equations/math intact
- [ ] All figure/table references correct
- [ ] Document compiles without errors

#### Step 5: Documentation Phase
```
a. Finalize Diagnostic Summary (3-4 bullet points)
b. Complete Revision Log (table or list format)
c. Generate Verification Checklist (all flagged items)
d. (Ultra Pass only) Write Style Rationale (1 page)
```

---

## PART IX: COMPARISON TABLE

### kmbart.tex Style vs. AI-Generated vs. Humanized

| **Characteristic** | **kmbart.tex** | **AI-Generated** | **Medium Pass** | **Extreme Pass** |
|--------------------|----------------|------------------|-----------------|------------------|
| **Sentence length** | 15-30 words (70%), varied | 20-25 words (85%), uniform | 15-30 words (70%), moderate variation | 10-35 words, high variation |
| **Sentence fragments** | None | None | Very rare (1-2) | Occasional (2-4 per section) |
| **"And/But" openers** | Rare | None | Few (1-2) | Several (4-6) |
| **Rhetorical questions** | None/very rare | None | None/very rare | Very rare (if helpful) |
| **Flow** | Smooth, polished | Perfectly smooth | Mostly smooth, slight friction | Natural friction, varied |
| **Transitions** | Formal, explicit | Formulaic | Mixed formal + occasional conversational | Mixed, some reflective |
| **Hedging** | Minimal, confident | Generic "may/could" | Specific, evidence-bound | Specific, scope-anchored |
| **Personal voice** | "We" active voice | Passive/impersonal | Clear "We" presence | Distinct researcher voice |
| **AI words** | None | Heavy use (robust, leverage, paradigm) | Mostly removed | Aggressively removed |
| **Paragraph structure** | Varied entry points | Rigid Topic→Evidence | Moderately varied | Boldly varied |
| **Specificity** | High (exact numbers) | Often vague | High (added details) | Very high (all verified) |

---

## APPENDIX A: BANNED WORDS & REPLACEMENTS

### Tier 1: Always Replace
| AI Word | kmbart.tex Alternative |
|---------|------------------------|
| tapestry | combination, integration, structure |
| delve | examine, investigate, explore, analyze |
| paradigm | approach, model, framework, method |
| realm | domain, area, field |
| landscape | field, domain, area, overview |
| groundbreaking | significant, substantial (with evidence) |
| cutting-edge | recent, current, advanced (if factual) |

### Tier 2: Context-Dependent (Replace if Hype)
| AI Word | kmbart.tex Alternative |
|---------|------------------------|
| robust/robustness | stable, consistent, reliable, effective, strong |
| leverage | use, apply, utilize, employ (technical OK) |
| novel | new, proposed (factual only) |
| foster | enable, support, facilitate, promote |
| underscore/underscores | demonstrate, show, indicate, emphasize |
| myriad/plethora | many, multiple, various, numerous |
| empower | enable, allow, support |
| pivotal | key, important, critical, essential |
| endeavor | effort, work, attempt, study |
| holistic | comprehensive (only if specific) |

### Tier 3: Academic Fluff (Remove or Replace)
| Fluff Phrase | Replacement Strategy |
|--------------|---------------------|
| comprehensive (vague) | Specify scope: "covering X, Y, and Z datasets" |
| state-of-the-art (hype) | "best performing," "highest accuracy" + numbers |
| significant (no stats) | Add p-value or effect size |
| substantial (vague) | Quantify: "15% improvement," "50% reduction" |

### kmbart.tex Preferred Terms

**Verbs**:
- propose, introduce, present, describe
- demonstrate, show, achieve, reach, obtain
- improve, enhance, boost (always with numbers)
- employ, use, adopt, leverage (technical only)
- conduct, perform, execute (for experiments)
- observe, find, note (for findings)

**Nouns**:
- model, architecture, framework, approach
- task, objective, goal
- dataset, corpus, collection
- performance, accuracy, results, metrics
- method, technique, strategy, procedure

**Adjectives** (use sparingly):
- effective, efficient (with evidence)
- significant (with statistics)
- comprehensive (with scope)
- preliminary (for early results)

**Transitions**:
- However, Nevertheless, In contrast, On the other hand
- Therefore, Consequently, Thus, As a result, Hence
- Furthermore, Moreover, Additionally, In addition
- For instance, For example, Specifically
- To ease this problem, To address this issue

---

## APPENDIX B: L2 ENGLISH (CHINESE L1) CHARACTERISTICS

### Subtle Patterns (Use Sparingly)

**Important**: These patterns should be **extremely subtle** and used **very sparingly**. The kmbart.tex style is highly polished, so overusing L2 markers would reduce quality.

#### 1. Direct Verb Preferences
```
Prefer: "conduct experiments" over "carry out experiments"
Prefer: "use the model" over "make use of the model"
Prefer: "achieve results" over "get results"
Prefer: "obtain performance" over "gain performance"
```

#### 2. Formal Connector Preferences
```
Slight preference for "therefore" over "so"
Use "thus" and "hence" comfortably
"In addition" over "also" in formal contexts
"However" over "but" in formal contexts
```

#### 3. Sentence Structure
```
Preference for explicit subjects (less ellipsis)
Clear agent-action-object order
Explicit transitions rather than implicit ones
Example: "We use X to achieve Y" rather than "X achieves Y"
```

#### 4. Article/Preposition Patterns
**Very subtle, use extremely rarely**:
```
Slightly stiffer constructions in informal notes only
Direct translations of Chinese academic phrases (barely noticeable)
Example: "conduct the experiment" vs. "run an experiment"
```

**Warning**: Do NOT introduce grammatical errors. These are stylistic preferences only, used to add very subtle authenticity.

---

## APPENDIX C: KMBART.TEX EXAMPLE TRANSFORMATIONS

### Example 1: Introduction Paragraph

**AI-Generated**:
```
The field of vision-language models has witnessed significant progress
in recent years, with numerous groundbreaking approaches demonstrating
robust performance across a myriad of tasks. However, despite these
substantial advances, the paradigm of leveraging multimodal
representations for commonsense generation remains largely unexplored.
This comprehensive work seeks to delve into this realm by proposing a
novel framework that fosters enhanced understanding of cross-modal
relationships.
```

**After Medium Pass (kmbart.tex style)**:
```
Vision-language models have advanced rapidly in recent
years~\cite{lxmert,vilbert,uniter,ernie-vil}. These models achieve
strong performance on understanding tasks such as Visual Question
Answering~\cite{vqa} and Image-Text Matching~\cite{it}. However,
applying multimodal representations to commonsense generation has
received limited attention. We propose a framework that improves
cross-modal understanding for this task.
```

**Changes Made**:
- Removed AI words: "witnessed significant," "groundbreaking," "robust," "myriad," "substantial," "paradigm," "leveraging," "realm," "comprehensive," "delve," "novel," "fosters"
- Added specific citations
- Broke into clearer sentences
- Used kmbart.tex verbs: "advanced," "achieve," "propose," "improves"
- Added specificity: task names, citation keys

---

### Example 2: Methodology Section

**AI-Generated**:
```
Our approach leverages a robust multi-layer architecture that may
potentially enable the model to learn comprehensive representations.
The paradigm employs a myriad of pretraining objectives that could
foster enhanced performance across diverse downstream tasks.
```

**After Low Pass (kmbart.tex style)**:
```
Our approach uses a multi-layer Transformer architecture that enables
the model to learn multimodal representations. We employ multiple
pretraining objectives to improve performance on downstream tasks
(Section 3.2).
```

**Changes Made**:
- "leverages" → "uses"
- "robust" → removed
- "may potentially enable" → "enables" (confident)
- "comprehensive" → "multimodal" (specific)
- "paradigm employs" → "We employ"
- "myriad of" → "multiple"
- "could foster enhanced" → "to improve"
- Added cross-reference

---

### Example 3: Results Section

**AI-Generated**:
```
The experimental results underscore that our novel approach achieves
state-of-the-art performance, demonstrating significant improvements
that may suggest the robustness of our proposed paradigm.
```

**After Medium Pass (kmbart.tex style)**:
```
Experimental results show that our model achieves state-of-the-art
performance on the VCG task~\cite{vcg}. Our model reaches a CIDER
score of 39.76, outperforming the previous best result of 39.13 (Table
4). This demonstrates the effectiveness of our proposed pretraining
tasks.
```

**Changes Made**:
- "underscore" → "show"
- "novel" → removed (kept in "our model")
- Added specific metrics (CIDER 39.76)
- "significant improvements" → specific comparison
- "may suggest the robustness" → "demonstrates the effectiveness"
- "proposed paradigm" → "proposed pretraining tasks" (specific)
- Added table reference

---

## APPENDIX D: PASS INTENSITY DECISION TREE

### When to Use Each Pass

```
START: Analyze AI-generated text

Is the text already highly polished and formal?
├─ YES → Consider Low Pass
│   └─ Are there only minor AI word choices to fix?
│       ├─ YES → Use Low Pass
│       └─ NO → Use Medium Pass
│
└─ NO → Continue

Are there moderate AI patterns (uniform sentences, some AI words)?
├─ YES → Consider Medium Pass
│   └─ Is this a formal methodology or results section?
│       ├─ YES → Use Medium Pass
│       └─ NO → Use Strong Pass
│
└─ NO → Continue

Are there heavy AI patterns (many AI words, formulaic structure)?
├─ YES → Consider Strong Pass
│   └─ Is maximum de-AI needed for detection resistance?
│       ├─ YES → Use Extreme or Ultra Pass
│       └─ NO → Use Strong Pass
│
└─ RARE: Use Extreme/Ultra Pass only for maximum humanization needs
```

### Section-Type Recommendations

| Section Type | Recommended Pass | Rationale |
|--------------|------------------|-----------|
| Abstract | Low | Very formal, concise |
| Introduction | Medium-Strong | Narrative, some flexibility |
| Related Work | Low-Medium | Formal, citation-heavy |
| Methodology | Low-Medium | Highly structured, precise |
| Results | Medium | Formal but narrative |
| Discussion | Strong | Interpretive, more voice |
| Conclusion | Medium-Strong | Summary + forward-looking |

---

## DOCUMENT METADATA

**Version**: 1.0
**Created**: 2025-10-02
**Last Updated**: 2025-10-02
**Based On**:
- kmbart.tex language analysis
- Humanization protocol from prompt.md
- Pass intensity presets (Low/Medium/Strong/Extreme/Ultra)

**Target Audience**:
- Master's/PhD students revising AI-assisted thesis writing
- Academic editors helping students improve AI-generated drafts
- Researchers ensuring AI-assisted work meets publication standards

**Scope**:
- Academic thesis writing in computer science / machine learning
- Formal scholarly writing (research papers, technical reports)
- English L2 (Chinese L1) academic voice

**Out of Scope**:
- Creative writing, journalism, blog posts
- Informal academic writing (emails, notes)
- Non-technical academic fields with different conventions

**Usage Notes**:
1. Always start with Diagnostic Audit (Part III)
2. Select appropriate Pass Intensity (Part VI)
3. Follow Practical Workflow (Part VIII)
4. Maintain LaTeX integrity throughout
5. Document all changes in Revision Log
6. Flag uncertain items in Verification Checklist

**Ethical Considerations**:
- This protocol is for **editing AI-assisted drafts**, not concealing AI authorship
- Users must disclose AI assistance per institutional policies
- All intellectual content must originate from the human researcher
- Factual claims must be verified against real data
- Citations must be accurate and verifiable

---

**END OF GUIDELINES DOCUMENT**