# Amoebanator V1.0 - Vignette Schema v2.0

Comprehensive reference for the Pydantic v2 schema underlying the 9-class
meningoencephalitis differential ML system. Schema locked at v2.0 as of
Subphase 1.1 closure (May 2026).

---

## Section 1: Overview

The vignette schema is the contract that every clinical case in the 270-vignette
corpus must satisfy before it enters training, calibration, or evaluation. The
schema is defined in `ml/schemas/vignette.py` as a hierarchy of 14 Pydantic v2
`BaseModel` classes, with cross-field validators enforcing class-conditional
clinical rules.

**Headline numbers**

- 14 sub-models, ~75 leaf fields
- 5 cross-field `model_validator` rules
- 9 `ClassLabel` enum values, 1 = PAM, 9 = NON_INFECTIOUS_MIMIC
- `schema_version` Literal-locked at "2.0"
- `extra="forbid"` at every model (no rogue fields permitted)
- Validation performance: P99 0.0265 ms per vignette (188x under 5 ms target)

**Public API**

```python
from ml.schemas import (
    ClassLabel, VignetteSchema,
    Demographics, History, ExposureHistory, VitalSigns,
    PhysicalExam, Labs, CSFProfile, Imaging,
    DxResult, DiagnosticTests,
    AdjudicationMetadata, LiteratureAnchor, Provenance,
)
```

**Files in this package**

| Path | Purpose |
|---|---|
| `ml/schemas/labels.py` | `ClassLabel` IntEnum + per-class citations |
| `ml/schemas/vignette.py` | All 14 sub-models + 5 validators |
| `ml/schemas/__init__.py` | Public re-exports (15 symbols) |
| `ml/schemas/export_json_schema.py` | JSON Schema export utility |
| `ml/schemas/SCHEMA_README.md` | This document |
| `schemas/vignette_schema_v2.0.json` | Exported JSON Schema (41 KB) |
| `scripts/vignettes/generate_subphase11_fixtures.py` | Reproducible fixture generator |
| `tests/schemas/` | 25 tests (24 PASSED, 1 SKIPPED for V1.5 migration) |
| `tests/schemas/fixtures/` | 9 canonical vignettes (1 per ClassLabel) |

---

## Section 2: Sub-model reference

For each sub-model: purpose, fields with constraints, citations for range bounds,
and validator (where applicable). Numeric ranges are enforced via Pydantic
`Field(ge=..., le=...)`; categorical fields via `Literal[...]`.

### 2.1 Demographics (5 fields)

Demographic context. Used downstream for stratified evaluation.

| Field | Type | Constraint | Justification |
|---|---|---|---|
| `age_years` | `int` | 0-110 | PAM bimodal age (Gharpure 2021); HSV-1 bimodal; NCC 20-40 |
| `sex` | `Literal["male", "female", "intersex"]` | required | PAM ~75% male; NMDAR ~80% female |
| `ethnicity` | `Optional[Literal[8 values]]` | optional | Hispanic/mestizo overrepresented in Balamuthia (Bravo PMC8760460) |
| `geography_region` | `Literal[10 values]` | required | 6 Peru regions + US South + Pakistan-Karachi + 2 fallbacks |
| `altitude_residence_m` | `Optional[int]` | 0-5000 | HACE risk threshold >2,500 m (WMS 2024 PMID 37833187) |

### 2.2 History (4 fields)

History of present illness.

| Field | Type | Constraint |
|---|---|---|
| `symptom_onset_to_presentation_days` | `float` | 0-180 |
| `chief_complaint` | `Literal[10 values]` | required |
| `prodrome_description` | `Optional[str]` | max 500 chars |
| `red_flags_present` | `List[Literal[8 values]]` | default `[]` |

PAM time course 1-12 d (MMWR 2025); TB 14-56 d; cryptococcal 7-84 d.

### 2.3 ExposureHistory (8 fields + 1 validator)

Exposure history including the PAM always-flag fields.

| Field | Type | Constraint |
|---|---|---|
| `freshwater_exposure_within_14d` | `bool` | **CRITICAL - required** for PAM always-flag rule |
| `freshwater_exposure_type` | `Optional[Literal[8 values]]` | optional |
| `altitude_exposure_within_7d_m` | `Optional[int]` | 0-6000 |
| `pork_consumption_or_taenia_contact` | `Optional[bool]` | NCC criterion (Del Brutto 2017) |
| `mosquito_endemic_area_exposure` | `Optional[bool]` | malaria/arboviral |
| `immunocompromise_status` | `Literal[10 values]` | required |
| `hiv_status` | `Literal[4 values]` | required |
| `cd4_count_cells_per_uL` | `Optional[int]` | 0-2000; required for cryptococcal+HIV+ via cross-validator |

**Validator `_freshwater_type_requires_exposure`:** if `freshwater_exposure_within_14d=True`,
then `freshwater_exposure_type` MUST be set (CDC PAM 2017 case definition).

### 2.4 VitalSigns (7 fields)

Vital signs at presentation.

| Field | Type | Constraint |
|---|---|---|
| `temperature_celsius` | `float` | 34.0-42.5 |
| `heart_rate_bpm` | `int` | 30-220 |
| `systolic_bp_mmHg` | `int` | 50-260 |
| `diastolic_bp_mmHg` | `int` | 30-160 |
| `glasgow_coma_scale` | `int` | 3-15 (Teasdale & Jennett 1974); WHO cerebral malaria GCS<11 |
| `oxygen_saturation_pct` | `Optional[int]` | 50-100 |
| `respiratory_rate_breaths_per_min` | `int` | 6-60 |

### 2.5 PhysicalExam (8 fields)

Examination findings.

| Field | Type | Constraint |
|---|---|---|
| `mental_status_grade` | `Literal[5 values]` | alert / confused / somnolent / stuporous / comatose |
| `neck_stiffness` | `bool` | required (van de Beek triad) |
| `kernig_or_brudzinski_positive` | `Optional[bool]` | low sensitivity but high specificity (Thomas CID 2002) |
| `focal_neurological_deficit` | `bool` | required |
| `cranial_nerve_palsy` | `Literal[6 values]` | CN VI palsy in TB meningitis |
| `skin_lesion_centrofacial_chronic` | `Optional[bool]` | 73% Peruvian Balamuthia (Bravo) |
| `petechial_or_purpuric_rash` | `bool` | 62-81% meningococcal (van de Beek 2008) |
| `papilledema_on_fundoscopy` | `Optional[bool]` | elevated ICP marker |

### 2.6 Labs (6 fields)

Peripheral lab values.

| Field | Type | Constraint |
|---|---|---|
| `wbc_blood_per_uL` | `int` | 100-80000 |
| `platelets_per_uL` | `int` | 5000-800000 (severe dengue/malaria <100k) |
| `alt_ast_U_per_L` | `Optional[int]` | 5-10000 |
| `crp_mg_per_L` | `Optional[float]` | 0-500 (>40 favors bacterial, Sormunen 1999) |
| `procalcitonin_ng_per_mL` | `Optional[float]` | 0-200 (>0.5 favors bacterial, Vikse 2015) |
| `serum_sodium_mEq_per_L` | `int` | 100-170 (<125 = severe symptomatic hyponatremia) |

### 2.7 CSFProfile (14 fields + 1 validator) - diagnostic core

The primary diagnostic discriminator across the 9 classes.

| Field | Type | Constraint | Reference |
|---|---|---|---|
| `opening_pressure_cmH2O` | `float` | 5-60 | >=25 in 60-80% HIV-cryptococcal (NIH OI) |
| `csf_wbc_per_mm3` | `int` | 0-50000 | IDSA Tunkel 2004 |
| `csf_neutrophil_pct` | `int` | 0-100 | bacterial 80-95% neutrophilic |
| `csf_lymphocyte_pct` | `int` | 0-100 | viral/TB/fungal lymphocytic |
| `csf_eosinophil_pct` | `int` | 0-100 (default 0) | >10% suggests parasitic (Del Brutto 2017) |
| `csf_glucose_mg_per_dL` | `int` | 0-200 | <34 mg/dL = >=99% bacterial specificity (Tunkel 2004) |
| `csf_protein_mg_per_dL` | `int` | 5-3000 | bacterial often >220; TB 100-500 |
| `csf_lactate_mmol_per_L` | `Optional[float]` | 0.5-20 | >3.5 favors bacterial (Sakushima 2011) |
| `csf_ada_U_per_L` | `Optional[float]` | 0-100 | >=10 U/L optimal TB cutoff (Ye TM&IH 2023) |
| `csf_crag_lfa_result` | `Optional[Literal]` | positive / negative / not_done | Williams CID 2015 (~100% sens HIV-cryptococcal) |
| `csf_wet_mount_motile_amoebae` | `Optional[Literal]` | positive / negative / not_done | N. fowleri (Balamuthia/Acanthamoeba usually negative) |
| `csf_xanthochromia_present` | `Optional[bool]` | required for SAH workup | Perry BMJ 2015 |
| `csf_rbc_per_mm3` | `Optional[int]` | 0-1,000,000 | SAH discrimination |
| `csf_rbc_decreasing_across_tubes` | `Optional[bool]` | traumatic vs SAH | tube 1 to 4 trend |

**Validator `_csf_differential_sums_to_100`:** when `csf_wbc_per_mm3 > 5`, the
sum of `neutrophil_pct + lymphocyte_pct + eosinophil_pct` MUST be in `[98, 102]`
(allowing rounding). Acellular CSF (WBC<=5) is exempt.

### 2.8 Imaging (4 fields)

Neuroimaging summary.

| Field | Type | Constraint |
|---|---|---|
| `imaging_modality` | `Literal[6 values]` | none / CT* / MRI* (MRI with FLAIR/DWI preferred per IDSA 2008) |
| `imaging_pattern` | `Optional[Literal[14 values]]` | class-discriminative pattern (HSV-1 mesial temporal, TB basal meningeal, GAE multiple ring-enhancing, etc.) |
| `imaging_finding_count` | `Optional[int]` | 0-50 (NCC staging, GAE multifocality) |
| `imaging_text_summary` | `Optional[str]` | max 1000 chars (radiologist impression) |

### 2.9 DxResult (5 fields)

Single diagnostic test result with sensitivity/specificity attribution.

| Field | Type | Constraint |
|---|---|---|
| `test_name` | `str` | 1-200 chars |
| `result` | `str` | 1-500 chars |
| `sensitivity_pct` | `Optional[float]` | 0-100 |
| `specificity_pct` | `Optional[float]` | 0-100 |
| `citation_pmid_or_doi` | `str` | 1-200 chars (required) |

### 2.10 DiagnosticTests (1 field)

Wrapper for an array of `DxResult` items. Includes Gram stain, blood/CSF
cultures, HSV-1/enterovirus PCR, FilmArray ME panel (PMID 31760115), Xpert
MTB/RIF Ultra (Cresswell CID 2020), EITB cysticercosis (Tsang 1989),
Balamuthia IFA, brain biopsy histology, malaria thick/thin smear, dengue NS1,
free-living amebae mNGS.

| Field | Type | Constraint |
|---|---|---|
| `results` | `List[DxResult]` | default `[]` |

### 2.11 AdjudicationMetadata (5 fields)

Physician adjudication metadata.

| Field | Type | Constraint |
|---|---|---|
| `adjudicator_ids` | `List[str]` | min 2 items (Cohen 1960; McHugh 2012 PMC3900052) |
| `cohen_kappa` | `float` | 0.0-1.0 (>=0.61 substantial per Landis & Koch; >=0.7 preferred clinical) |
| `disagreement_resolution` | `Optional[Literal]` | consensus_discussion / third_adjudicator / excluded |
| `anchoring_documentation` | `str` | 1-2000 chars |
| `inclusion_decision` | `Literal[3 values]` | include / exclude / hold_for_revision |

### 2.12 LiteratureAnchor (3 fields + 1 validator)

Peer-reviewed literature anchor for vignette clinical fidelity.

| Field | Type | Constraint |
|---|---|---|
| `anchor_type` | `Literal[8 values]` | case_report / cohort / rct / surveillance / etc. |
| `pmid` | `Optional[str]` | regex `^\d{7,9}$` |
| `doi` | `Optional[str]` | regex `^10\.\d{4,9}/.+$` |

**Validator `_at_least_one_id`:** at least one of `pmid` OR `doi` MUST be
non-`None`. Both `None` is invalid (citation traceability requirement).

### 2.13 Provenance (5 fields)

Vignette generation provenance for audit trail and reproducibility.

| Field | Type | Constraint |
|---|---|---|
| `generation_timestamp_utc` | `datetime` | ISO-8601 UTC |
| `generator_model_identifier` | `str` | 1-200 chars |
| `prompt_hash_sha256` | `str` | regex `^[a-f0-9]{64}$` (NIST FIPS 180-4) |
| `schema_version` | `Literal["2.0"]` | locked |
| `inclusion_decision_rationale` | `str` | max 1000 chars (carries `IMPUTED_FROM_LITERATURE` markers) |

### 2.14 VignetteSchema (top-level, 17 fields + 2 validators)

Top-level container.

| Field | Type | Constraint |
|---|---|---|
| `schema_version` | `Literal["2.0"]` | locked |
| `case_id` | `str` | 1-200 chars |
| `ground_truth_class` | `ClassLabel` | 1-9 |
| `demographics` | `Demographics` | required |
| `history` | `History` | required |
| `exposure` | `ExposureHistory` | required |
| `vitals` | `VitalSigns` | required |
| `exam` | `PhysicalExam` | required |
| `labs` | `Labs` | required |
| `csf` | `CSFProfile` | required |
| `imaging` | `Imaging` | required |
| `diagnostic_tests` | `DiagnosticTests` | required |
| `adjudication` | `AdjudicationMetadata` | required |
| `literature_anchors` | `List[LiteratureAnchor]` | min 1 |
| `provenance` | `Provenance` | required |
| `narrative_es` | `Optional[str]` | max 4000 chars |
| `narrative_en` | `Optional[str]` | max 4000 chars |

Validators: see Section 4 (`_pam_always_flag_rule`, `_cryptococcal_cd4_required_when_hiv`).

---

## Section 3: ClassLabel reference (per-class anchor)

For each of 9 classes: enum value, canonical anchor, optional Peru companion,
geographic relevance score for Peru deployment, adjudicator notes.

### 3.1 ClassLabel.PAM (1) - Primary Amebic Meningoencephalitis

- **Canonical anchor:** PMID 40146665 (MMWR 2025) + CDC PAM 2017 case definition
- **DOI:** 10.15585/mmwr.mm7410a2
- **Peru relevance:** medium - Peru reports rare; primary literature US/Pakistan
- **Adjudicator notes:** confirm always-flag rule wording; verify splash pad as
  acceptable freshwater exposure type per MMWR 2025 Pulaski County Arkansas case

### 3.2 ClassLabel.BACTERIAL (2) - Acute bacterial meningitis

- **Canonical anchor:** PMID 15509818, DOI 10.1056/NEJMoa040845
  - van de Beek D, et al. Clinical features and prognostic factors in adults
    with bacterial meningitis. NEJM 2004;351(18):1849-1859. (Netherlands n=696)
- **Peru companion (pediatric only):** PMID 27831604
  - Castillo ME, Solís S, Verne E, et al. RPMESP 2016;33(3):425-431
- **Modern epidemiology backup:** PMID 34036322
  - Koelman DLH, Brouwer MC, Ter Horst L, **Bijlsma MW**, van der Ende A,
    van de Beek D. Pneumococcal Meningitis in Adults. CID 2022;74(4):657-667
  - **D.1 Fix 4 applied:** Bijlsma MW now included in author string
- **Peru relevance:** high (universal disease); but no Peru ADULT cohort exists
- **Adjudicator notes:** confirm acceptability of pediatric Peru companion ONLY;
  flag if HNDM/HCH/Almenara local extraction should be commissioned in Phase 2

### 3.3 ClassLabel.VIRAL (3) - Viral meningoencephalitis

- **Canonical Peru anchor:** PMID 26733400, DOI 10.1017/S0950268815003222
  - **Montano SM, Mori N, Nelson CA, Ton TGN, Celis V**, et al.
    Epidemiol Infect 2016;144(8):1673-1678. HSV encephalitis 5 Peru cities
  - **D.1 Fix 3 applied:** corrected from "Becerra" to Montano SM first author
- **Peru relevance:** high - Lima cohort directly Peruvian
- **Adjudicator notes:** confirm HSV-1 representativeness; consider HSV-2,
  enterovirus, arboviral subclasses for Phase 2 expansion

### 3.4 ClassLabel.TUBERCULOUS (4) - TB meningitis

- **Canonical Peru anchor:** PMID 30611205, DOI 10.1186/s12879-018-3633-4
  - Soria J, et al. BMC Infect Dis 2018. HNDM Lima TB meningitis cohort
- **Pathophysiology reference:** Marais S, et al. Lancet Infect Dis 2010;10(11):803-812
  - DOI 10.1016/S1473-3099(10)70138-9
- **Peru relevance:** high - directly anchored to Hospital Nacional Dos de Mayo Lima
- **Adjudicator notes:** confirm CSF ADA cutoff (>=10 U/L per Ye TM&IH 2023)
  and basal meningeal enhancement + hydrocephalus as canonical imaging pattern

### 3.5 ClassLabel.CRYPTOCOCCAL_FUNGAL (5) - Cryptococcal/fungal meningitis

- **Canonical anchor:** PMID 35320642, DOI 10.1056/NEJMoa2111904
  - Jarvis JN, et al. Single-Dose Liposomal Amphotericin B Treatment for
    Cryptococcal Meningitis. AMBITION-cm trial. NEJM 2022;386(12):1109-1120
- **Peru companion:** PMID 28355252, DOI 10.1371/journal.pone.0174459
  - Concha-Velasco F, González-Lagos E, Seas C, Bustamante B. Factors
    associated with early mycological clearance in HIV-associated cryptococcal
    meningitis. PLoS One 2017;12(3):e0174459. Hospital Cayetano Heredia Lima
- **Peru relevance:** high - Peru companion directly relevant
- **Adjudicator notes:** confirm CD4-required-when-HIV+ validator; document
  5-FC unavailability at HCH per Concha-Velasco 2017 limitation

### 3.6 ClassLabel.GAE (6) - Granulomatous amebic encephalitis (Acanthamoeba/Balamuthia)

- **Canonical Peru single-patient:** PMID 20550438, DOI 10.1086/653609
  - **Martínez DY, Seas C, Bravo F, Legua P, Ramos C, Cabello AM, Gotuzzo E.**
    Successful Treatment of Balamuthia mandrillaris Amoebic Infection with
    Extensive Neurological and Cutaneous Involvement. CID 2010;51(2):e7-e11
  - **D.1 Fix 1 applied:** corrected from PMID 20550458 (off-by-one) to 20550438
- **Master Peru series:** PMID 35059659, DOI 10.1016/j.jdin.2021.11.005
  - Alvarez P, Torres-Cabala C, Gotuzzo E, Bravo F. **JAAD International**
    2022;6:51-58 (NOT JAAD Case Reports)
  - **D.1 Fix 2 applied:** corrected from DOI 10.1016/j.jdcr.2021.11.022 (JAAD
    Case Reports) to DOI 10.1016/j.jdin.2021.11.005 (JAAD International)
- **Peru relevance:** highest - Peru is global Balamuthia hotspot per UPCH/HCH
- **Adjudicator notes:** verify centrofacial skin lesion preceding CNS by
  median 15 months (Bravo PMC8760460 cohort); confirm Hispanic ethnicity
  framing as epidemiologic observation, not predictor

### 3.7 ClassLabel.NEUROCYSTICERCOSIS (7) - Acute neurocysticercosis

- **Canonical Peru anchor:** PMID 38003778, DOI 10.3390/pathogens12111313
  - Allen et al. Pathogens 2023;12(11):1313. Tumbes community-based NCC + epilepsy
- **Diagnostic criteria:** Del Brutto OH, et al. J Neurol Sci 2017;372:202-210
  - PMID 28017213 (revised diagnostic criteria, scolex within cyst absolute)
- **Peru relevance:** highest - 38% community-acquired epilepsy in Tumbes is NCC
- **Adjudicator notes:** confirm scolex-within-cyst as the absolute criterion;
  validate Taenia/pork exposure field for endemic-area risk stratification

### 3.8 ClassLabel.CEREBRAL_MALARIA_OR_SEVERE_ARBO (8) - Cerebral malaria + severe arboviral CNS

**Critical paradigm shift:** Peru malaria is 81% P. vivax (INS surveillance).
African pediatric P. falciparum cerebral malaria is biologically and clinically
distinct. Peru-anchor papers are the primary reference; African P. falciparum
literature is comparative-only.

- **Primary Peru anchor:** PMID 36477327, DOI 10.17843/rpmesp.2022.392.10739
  - Paredes-Obando M, et al. Plasmodium vivax cerebral malaria with pancytopenia
    in the Peruvian Amazon: case report. RPMESP 2022;39(2):241-244. Hospital
    Regional de Loreto, Iquitos
- **Comparative pathophysiology only:** Idro R, et al. Lancet Neurol 2005;4(12):827-840
  - PMID 15005962 - kept as physiology reference, NOT vignette anchor
- **Peru relevance:** highest - Loreto/Madre de Dios endemic
- **Adjudicator notes:** validate P. vivax vs P. falciparum distinction in
  fixture demographic + parasitemia% representation

### 3.9 ClassLabel.NON_INFECTIOUS_MIMIC (9) - Non-infectious mimics

Heterogeneous catch-all: anti-NMDAR encephalitis, HACE, severe migraine,
hyponatremia <125, PRES/RCVS, SAH.

- **Canonical anchor (anti-NMDAR):** PMID 25400967, DOI 10.1155/2014/868325
  - Keller S, et al. Anti-NMDA receptor encephalitis case. Israel
- **Diagnostic criteria:** Graus F, et al. Lancet Neurol 2016;15(4):391-404
  - DOI 10.1016/S1474-4422(15)00401-9 (autoimmune encephalitis criteria)
- **Other mimic references:** Hinchey J, et al. NEJM 1996;334(8):494-500 (PRES)
- **Peru relevance:** medium - HACE highly relevant for Cusco/Puno deployment
- **Adjudicator notes:** confirm subtype enumeration is sufficient or whether
  Phase 2 should split into NMDAR/HACE/PRES/SAH/HYPONATREMIA distinct labels

---

## Section 4: Cross-field validators (5)

Implementation: Pydantic v2 `@model_validator(mode="after")` on the relevant
sub-model. Tests verify each fires correctly with appropriate error messages.

### 4.1 `_freshwater_type_requires_exposure` (ExposureHistory)

- **Trigger:** `freshwater_exposure_within_14d=True` AND `freshwater_exposure_type is None`
- **Action:** raise ValueError
- **Rationale:** CDC PAM 2017 case definition requires the type of freshwater
  exposure to be documented when exposure is reported. Allowing `True` with
  unspecified type would defeat the purpose of the always-flag rule.
- **Test:** `test_pam_class_requires_freshwater_exposure` (smoke test 4 partially)

### 4.2 `_csf_differential_sums_to_100` (CSFProfile)

- **Trigger:** `csf_wbc_per_mm3 > 5` AND `(neutrophil + lymphocyte + eosinophil) not in [98, 102]`
- **Action:** raise ValueError with explicit cell counts
- **Rationale:** IDSA Tunkel 2004 reference. Acellular CSF (WBC<=5) is exempt.
  Allows +/-2 percentage points for rounding artifacts in clinical reporting.
- **Test:** `test_csf_differential_must_sum_to_100` (smoke test 5)

### 4.3 `_at_least_one_id` (LiteratureAnchor)

- **Trigger:** `pmid is None` AND `doi is None`
- **Action:** raise ValueError
- **Rationale:** Citation traceability is non-negotiable. Every literature
  anchor must have at least one machine-resolvable identifier.
- **Test:** `test_literature_anchor_requires_pmid_or_doi` (smoke test 6)

### 4.4 `_pam_always_flag_rule` (VignetteSchema)

- **Trigger:** `ground_truth_class == ClassLabel.PAM` AND `exposure.freshwater_exposure_within_14d is False`
- **Action:** raise ValueError
- **Rationale:** Clinical safety floor. PAM without documented freshwater
  contact within 14 days is implausible per CDC 2017 case definition. Any
  vignette claiming PAM ground truth without the exposure cannot be a valid
  training/calibration sample.
- **Test:** `test_pam_class_requires_freshwater_exposure` (smoke test 4)

### 4.5 `_cryptococcal_cd4_required_when_hiv` (VignetteSchema)

- **Trigger:** `ground_truth_class == ClassLabel.CRYPTOCOCCAL_FUNGAL` AND
  `exposure.hiv_status in (positive_on_art, positive_not_on_art)` AND
  `exposure.cd4_count_cells_per_uL is None`
- **Action:** raise ValueError
- **Rationale:** WHO 2022 Advanced HIV guidelines + Ford CID 2018 PMC5850628.
  CD4 stratification (<100 cells/uL = high CrAg risk) is core to cryptococcal
  triage. A cryptococcal+HIV+ vignette without CD4 is incomplete.
- **Test:** indirectly validated by `valid_cryptococcal_fixture.json` which
  populates `cd4_count_cells_per_uL=38`

---

## Section 5: Documented limitations

### 5.1 Class 2 (Bacterial meningitis) - no Peru adult cohort

No large Peruvian adult community-acquired pyogenic bacterial meningitis cohort
exists in indexed literature (PubMed, SciELO, RPMESP) as of May 2026. Available
Peru evidence is pediatric (Castillo 2016 PMID 27831604; Marin-Portocarrero
2022 PMID 36888810) or single-case reports.

- **Master anchor used:** van de Beek 2004 PMID 15509818 (Netherlands n=696)
  as gold-standard prognostic-pathophysiology reference
- **Peru pediatric companion:** Castillo ME et al. RPMESP 2016;33(3):425-431
- **Modern epidemiology backup:** Koelman 2022 PMID 34036322 (Netherlands)
- **Adjudicator action:** confirm acceptability OR commission HNDM/HCH/Almenara
  local data extraction in Phase 2 supplement

### 5.2 Class 5 (Cryptococcal) - no Peru AMBITION-equivalent

No Peru randomized controlled trial of liposomal amphotericin B + 5-FC dual
therapy. Concha-Velasco 2017 explicitly documents 5-FC unavailability at HCH.

- **Master anchor:** Jarvis AMBITION-cm PMID 35320642 (sub-Saharan Africa RCT)
- **Peru companion:** Concha-Velasco F, González-Lagos E, Seas C, Bustamante B.
  PLoS One 2017 PMID 28355252 (Hospital Cayetano Heredia Lima)
- **Real Peru constraint:** induction therapy options limited by 5-FC supply
- **Adjudicator action:** acknowledge reality in fixture provenance + preprint
  limitations section

### 5.3 Class 8 (Cerebral malaria) - Peru P. vivax paradigm replaces African P. falciparum

Peru malaria is dominated by P. vivax (~81% per INS surveillance). African
pediatric P. falciparum cerebral malaria literature (Idro et al.) is
biologically and clinically distinct (different rosetting, different sequestration,
different age distribution, different mortality profile).

- **Primary Peru anchor:** Paredes-Obando 2022 PMID 36477327 (Loreto, Iquitos)
- **Comparative pathophysiology only:** Idro 2005 PMID 15005962 (Uganda) -
  retained as physiology reference, NOT vignette anchor
- **Implication:** P. vivax cerebral involvement is rarer but documented in
  Peruvian Amazon. Fixture reflects this Peru-relevant paradigm.

---

## Section 6: Citation list (17 anchors, all corrections applied)

Numbered list of all anchor citations across the 9 classes. Class assignment
in brackets.

1. **CDC** [1: PAM]. Primary Amebic Meningoencephalitis (PAM): 2017 case definition. ndc.services.cdc.gov
2. **PMID 40146665** [1: PAM]. MMWR 2025;74(10). Splash pad PAM Pulaski County Arkansas. DOI 10.15585/mmwr.mm7410a2
3. **PMID 15509818** [2: BACTERIAL]. van de Beek D, et al. NEJM 2004;351(18):1849-1859. DOI 10.1056/NEJMoa040845
4. **PMID 27831604** [2: BACTERIAL Peru companion]. Castillo ME, et al. RPMESP 2016;33(3):425-431
5. **PMID 34036322** [2: BACTERIAL backup]. Koelman DLH, Brouwer MC, Ter Horst L, **Bijlsma MW**, van der Ende A, van de Beek D. CID 2022;74(4):657-667
6. **PMID 26733400** [3: VIRAL]. **Montano SM, Mori N, Nelson CA, Ton TGN, Celis V**, et al. Epidemiol Infect 2016;144(8):1673-1678. DOI 10.1017/S0950268815003222
7. **PMID 30611205** [4: TUBERCULOUS]. Soria J, et al. BMC Infect Dis 2018. DOI 10.1186/s12879-018-3633-4
8. **DOI 10.1016/S1473-3099(10)70138-9** [4: TUBERCULOUS pathophysiology]. Marais S, et al. Lancet Infect Dis 2010;10(11):803-812
9. **PMID 35320642** [5: CRYPTOCOCCAL]. Jarvis JN, et al. AMBITION-cm. NEJM 2022;386(12):1109-1120. DOI 10.1056/NEJMoa2111904
10. **PMID 28355252** [5: CRYPTOCOCCAL Peru companion]. Concha-Velasco F, et al. PLoS One 2017;12(3):e0174459. DOI 10.1371/journal.pone.0174459
11. **PMID 20550438** [6: GAE Peru single-patient]. Martínez DY, Seas C, Bravo F, Legua P, Ramos C, Cabello AM, Gotuzzo E. CID 2010;51(2):e7-e11. DOI 10.1086/653609 (D.1 Fix 1: corrected from PMID 20550458)
12. **PMID 35059659** [6: GAE master Peru series]. Alvarez P, Torres-Cabala C, Gotuzzo E, Bravo F. **JAAD International** 2022;6:51-58. DOI 10.1016/j.jdin.2021.11.005 (D.1 Fix 2: corrected from JAAD Case Reports DOI)
13. **PMID 38003778** [7: NCC Peru]. Allen et al. Pathogens 2023;12(11):1313. DOI 10.3390/pathogens12111313
14. **PMID 28017213** [7: NCC criteria]. Del Brutto OH, et al. J Neurol Sci 2017;372:202-210
15. **PMID 36477327** [8: CEREBRAL_MALARIA Peru]. Paredes-Obando M, et al. RPMESP 2022;39(2):241-244. DOI 10.17843/rpmesp.2022.392.10739
16. **PMID 25400967** [9: NMDAR case]. Keller S, et al. Case Rep Psychiatry 2014. DOI 10.1155/2014/868325
17. **DOI 10.1016/S1474-4422(15)00401-9** [9: NMDAR criteria]. Graus F, et al. Lancet Neurol 2016;15(4):391-404

---

## Section 7: Adjudicator review request

This SCHEMA_README.md is intended for forwarding to the physician collaborator
network for clinical-fidelity review. Specific questions per class:

- **Class 1 (PAM):** Confirm freshwater always-flag rule is appropriate clinical
  safety policy. Acceptable to enforce as schema-level validator?
- **Class 2 (Bacterial):** Confirm Castillo 2016 Peru pediatric companion is
  acceptable given absence of adult cohort. Should Phase 2 commission HNDM
  local extraction?
- **Class 3 (Viral HSV-1):** Confirm Montano 2016 Peru cohort is appropriate
  representative anchor.
- **Class 4 (TBM):** Confirm Soria 2018 HNDM cohort representative case
  construction approach. CN VI palsy + basal meningeal enhancement + ADA>=10
  combination acceptable?
- **Class 5 (Cryptococcal):** Confirm Concha-Velasco 2017 + AMBITION-cm
  dual-anchor strategy. Acknowledge 5-FC unavailability in fixture provenance
  acceptable?
- **Class 6 (GAE Balamuthia):** Confirm Martínez 2010 single-patient + Alvarez
  2022 series pairing. Hispanic ethnicity overrepresentation framed as
  epidemiologic observation rather than predictor - acceptable framing?
- **Class 7 (NCC):** Confirm Allen 2023 Tumbes cohort representative case.
  Scolex-within-cyst as absolute criterion sufficient?
- **Class 8 (Cerebral malaria):** Confirm Peru P. vivax paradigm
  (Paredes-Obando) replaces African P. falciparum (Idro). Comparative-only
  use of Idro acceptable?
- **Class 9 (Anti-NMDAR):** Confirm Graus 2016 criteria + Keller 2014 case
  pairing. Should Phase 2 split NON_INFECTIOUS_MIMIC into NMDAR / HACE / PRES /
  SAH / HYPONATREMIA distinct labels?

**Expected reviewer turnaround:** 2 weeks. Does NOT block code progression to
Subphase 1.2 (data ingestion pipeline). Reviewer feedback can be incorporated
in a subsequent schema patch (v2.1) without breaking v2.0 contracts.

---

## Section 8: Reproducibility

- **Schema source:** `ml/schemas/vignette.py` (~860 lines, 14 classes, 5 validators)
- **Generator script:** `scripts/vignettes/generate_subphase11_fixtures.py` (~750 lines)
- **JSON Schema export:** `schemas/vignette_schema_v2.0.json` (41 KB, Draft 2020-12 compatible)
- **Test suite:** `tests/schemas/` (24 PASSED + 1 SKIPPED)
- **Baseline:** 1,371 project tests passing as of `v2.0-schema-locked` tag

**To regenerate fixtures from scratch:**

```bash
python -m scripts.vignettes.generate_subphase11_fixtures
```

**To re-export JSON Schema:**

```bash
python -m ml.schemas.export_json_schema
```

**To run full schema test suite:**

```bash
python -m pytest tests/schemas/ -v
```

**Performance benchmark (P99 validation latency):** 0.0265 ms (188x under 5 ms
target). At training-loop scale (e.g., 270 vignettes x 30 epochs = 8,100
validations per training run) total validation overhead is ~215 ms.

---

**Schema status:** locked at v2.0. Subphase 1.1 closure as of May 2026.
Tag: `v2.0-schema-locked`.
