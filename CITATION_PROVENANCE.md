# SOAR-EU — Citation Provenance & Audit

A reference-by-reference record for both papers: the **gist** of each cited work, **how the paper uses it**, and its **verification status**. Intended as a review aid (e.g. for co-authors) and a submission-readiness record.

**Verification basis.** Every reference was checked for bibliographic existence (authors/year/title/venue) via web sources; quantitative claims were checked against the regenerated analysis macros/tables; historical and institutional claims were cross-checked against the scholarly literature; licenses were checked against the data providers' published terms. Status: ✅ verified and appropriately used · ⚠️ used with a noted caveat · 🔧 an issue was found and corrected this round.

---

## Data paper (24 references)

1. **`pesaresi2023`** — Pesaresi & Politis (2023), *GHS-BUILT-S R2023A* built-up surface grid (JRC). Global built-up-surface raster from the GHSL programme. *Used* in Background as an example of large-sample comparative built-environment data ("GHSL global built-up surface mapping"). DOI resolves to the correct JRC record. ✅

2. **`boeing2020`** — Boeing (2020), "A multi-scale analysis of 27,000 urban street networks," *Env. Plan. B* 47(4). Computational analysis of every US urban street network at multiple scales. *Used* as related work on large-sample street-network morphology. Title/volume/pages confirmed. ✅

3. **`fleischmann2022`** — Fleischmann et al. (2022), "Methodological foundation of a numerical taxonomy of urban form," *Env. Plan. B* 49(4). A *method* for numerically classifying urban form, demonstrated on Prague and Amsterdam. *Used* as related work; the in-text descriptor was corrected from "European urban form" to "urban form" (it is a methods paper, not a continental survey). 🔧

4. **`yap2023`** — Yap & Biljecki (2023), "A global feature-rich network dataset of cities," *Sci. Data* 10:667. The "Urbanity" dataset of 50 cities in 29 countries. *Used* as a related comparator ("feature-rich network dataset for 50 cities") — the 50-city figure is confirmed. ✅

5. **`simons2023`** — Simons (2023), "The cityseer Python package," *Env. Plan. B* 50(5). The network-based pedestrian-scale analysis engine. *Used* as the computational tool underlying all SOAR-EU metrics. ✅

6. **`simons2021thesis`** — Simons (2021), UCL PhD thesis. Pedestrian-scale integration of networks, land use and demographics across 931 GB towns/cities. *Used* as the methodological precursor. The "931" figure is corroborated via the thesis's associated outputs (the UCL record itself was access-blocked). ✅

7. **`openshaw1984`** — Openshaw, *The Modifiable Areal Unit Problem*, CATMOG 38. Establishes the MAUP's scale and zonation effects. *Used* in Value-of-the-Data to justify providing multiple spatial scales (relationships at one scale need not hold at another) — the scale effect supports this. ✅

8. **`apparicio2008`** — Apparicio et al. (2008), *Int. J. Health Geogr.* 7:7. Compares distance types (Euclidean vs network) and aggregation error in accessibility measurement. *Used* to justify network-based accessibility over Euclidean buffers. ✅

9. **`ghsl2024`** — EC JRC, *GHS Urban Centre Database R2024A*. Defines urban centres via the Degree of Urbanisation (DEGURBA). *Used* for study-area definition; the DEGURBA definition in the text was corrected to include the built-up-surface limb (≥1,500 inhab/km² **or** ≥50% built-up). 🔧

10. **`overture2026`** — Overture Maps Foundation, Release 2026-05-20.0. Source for street networks, POIs, and building footprints. *Used* throughout as the primary dynamic data source; the release exists and is correctly dated. ✅

11. **`eea2021ua`** — Copernicus *Urban Atlas 2021*. Land-cover/use blocks and green-space classes. *Used* for block morphology and green-space derivation; the class count was corrected from 27 (2012/2018 nomenclature) to 28 (2021: 19 urban + 9 rural). 🔧

12. **`eea2021stl`** — Copernicus *Street Tree Layer 2021*. Tree-canopy polygons within Urban Atlas FUAs. *Used* for tree-canopy proximity. ✅

13. **`eea2012bh`** — Copernicus *Building Height 2012*. 10 m building-height raster. *Used* to sample building heights (→ volume, form factor). ✅

14. **`eurostat2024`** — Eurostat *Census 2021 Population Grid*. 1 km² INSPIRE demographic grid. *Used* for interpolated census demographics. ✅

15. **`hill1973`** — Hill (1973), *Ecology* 54(2). Defines Hill diversity numbers (q=0 richness; q=1 exp-Shannon; q=2 inverse-Simpson). *Used* for mixed-use land-use diversity; the q=0/1/2 definitions match exactly. ✅

16. **`fleischmann2019`** — Fleischmann (2019), *momepy*, *JOSS* 4(43). Urban morphometrics toolkit. *Used* to compute building/block morphometrics. ✅

17. **`haupt2010`** — Berghauser Pont & Haupt, *Spacematrix* (NAi). The density framework (GSI, FSI, OSR, L). *Used* for the Spacematrix block indicators; the formulas (OSR=(1−GSI)/FSI, L=FSI/GSI) match the source. ✅

18. **`herfort2023`** — Herfort et al. (2023), *Nat. Commun.* 14:3985. Global OSM building completeness across 13,189 agglomerations; Europe & Central Asia highest (71%). *Used* for building-provenance context; **the 13,189 and 71% figures are verified**, and a separate cadastral-imports claim that had been mis-attributed to Herfort was removed. 🔧

19. **`barringtonleigh2017`** — Barrington-Leigh & Millard-Ball (2017), *PLoS ONE* 12(8). Global OSM road map >80% complete; most European countries near-complete. *Used* for road-coverage context. ✅

20. **`minghini2024`** — Minghini et al. (2024), *ISPRS Archives*. Compares pan-European open building-footprint datasets; Microsoft ML footprints show the poorest geometric overlap. *Used* for the ML-geometry-mismatch point; the country-level generalization was softened to mark it as our own data observation. 🔧

21. **`sirene2026`** — INSEE *SIRENE* business registry. *Used* as the French reference registry in the POI source-substitution validation. License (Licence Ouverte / Open Licence 2.0, Etalab) verified. ✅

22. **`bag2026`** — Kadaster *BAG* building/address register. *Used* as the Dutch reference registry. Public-domain license verified; the footnote was softened to note that portal labels vary between CC Public Domain Mark 1.0 and CC0 1.0. 🔧

23. **`abdeldayem2026`** — Abdeldayem et al. (2026), *Env. Plan. B*. Automated vs hybrid street-network modelling for centrality/accessibility. *Used* for the network-cleaning procedure. Verified (published March 2026; DOI resolves). ✅

24. **`soar2026`** — Simons et al., *SOAR-EU* dataset, Zenodo. The deposited dataset itself. *Used* in Data Availability. Self-deposit; DOI populated (10.5281/zenodo.18961227). ⚠️ verify the deposit is public before submission.

---

## Atlas (38 references)

1. **`angel2016atlas`** — Angel et al. (2016), *Atlas of Urban Expansion*. Satellite-based mapping of urban extent and growth worldwide. *Used* as related work in large-sample comparative morphology. ✅

2. **`araldi2019street`** — Araldi & Fusco (2019), *Env. Plan. B* 46(7). Street-based, pedestrian-perspective urban-fabric analysis. *Used* for the move toward integrating across spatial-feature families. ✅

3. **`barringtonleigh2017world`** — Barrington-Leigh & Millard-Ball (2017), *PLoS ONE*. OSM road coverage largely complete across urban Europe. *Used* in Data Sources to support road-network reliability. ✅

4. **`ballantyne2024overture`** — Ballantyne & Berragan (2024), *Env. Plan. B*. Evaluates Overture Places (POI) quality against reference data. *Used* to flag systematic POI undercounting. ✅

5. **`berghauser2019systematic`** — Berghauser Pont et al. (2019), "The spatial distribution and frequency of street, plot and building types across five European cities," *Env. Plan. B* 46(7). Quantitative profiling/comparison of urban types. *Used* (L84) to motivate an intentionally simple classification (data-driven morphological clustering is hard). The bib entry was corrected — it previously carried a fabricated title and a wrong author. 🔧 ⚠️ confirm the corrected source fits the "clustering is a challenge" framing, or pair with a clustering-methods reference.

6. **`berghauser2010spacematrix`** — Berghauser Pont & Haupt (2010), *Spacematrix*. The FSI/GSI density framework. *Used* for the Intensity axis (FSI) and the density-as-condition argument. ✅

7. **`berghauser2021density`** — Berghauser Pont & Haupt (2021), *Buildings & Cities* 2(1). Density as a multi-indicator condition across scales. *Used* to argue density components carry independent information. ✅

8. **`busquets2005barcelona`** — Busquets (2005), *Barcelona: The Urban Evolution of a Compact City*. *Used* (L140) for the ensanche account: deep-lot perimeter blocks whose courtyards were progressively filled as ordinances raised buildable volume. ⚠️ supported; note "floor-area ratio" is a modern shorthand (period instruments were height/depth/volume/coverage rules) and Cerdà's original blocks were open on 2–3 sides.

9. **`boeing2019urban`** — Boeing (2019), *Applied Network Science* 4:67. Street-network orientation/entropy distinguishes planned vs organic cities. *Used* (L58, L90) for the irregularity rationale. ✅

10. **`boeing2020multi`** — Boeing (2020), *Env. Plan. B* 47(4). 27,000-network multi-scale analysis. *Used* as related work. ✅

11. **`caniggia2001architectural`** — Caniggia & Maffei (2001), *Architectural Composition and Building Typology*. The Muratorian process-typology tradition (building types aggregating along routes). *Used* (L58) for the building-typology tradition. The separate L160 use (perimeter block as *the* characteristic European form) was re-attributed to Panerai/Castex, which is the correct source for that thesis. 🔧

12. **`dedecker2008facets`** — De Decker (2008), *J. Housing Built Environ.* 23(3). Belgian housing culture, weak planning, and ribbon development. *Used* (L144) for *lintbebouwing* — an apt, well-supported citation. ✅

13. **`diefendorf1993wake`** — Diefendorf (1993), *In the Wake of War*. Documents divergent post-WWII reconstruction strategies across German cities. *Used* (L144) to explain Germany's wide spread of continuity values. ✅

14. **`dibble2019origin`** — Dibble et al. (2019), *Env. Plan. B* 46(4). Morphometric foundations of urban-form evolution; orientation separates development periods. *Used* (L58, L90) for the irregularity axis. ✅

15. **`dovey2014urban`** — Dovey & Pafka (2014), *Urban Design Int'l* 19. The urban density assemblage (multiple independent density measures). *Used* (L86) to support component-independence of density. ✅

16. **`fleischmann2022morphological`** — Fleischmann et al. (2022), *Env. Plan. B* 49(4). Numerical taxonomy of urban form (method). *Used* as related work (L58). ✅

17. **`faludi1994rule`** — Faludi & van der Valk (1994), *Rule and Order: Dutch Planning Doctrine in the Twentieth Century*. The doctrine of 20th-c. Dutch (plan-led) planning. *Used* (L144) for the Dutch plan-led tradition. This citation was re-targeted to the plan-led claim (where it fits) and removed from the polder-drainage claim (where it did not); the polder statement was softened from "imposed an orthogonal template" to "echoed in." 🔧

18. **`florczyk2019ghs`** — Florczyk et al. (2019), *GHS Urban Centre Database* (JRC). The urban-centre boundary definition. *Used* (L62) for the GHS-UCDB extents. ✅

19. **`hamaina2012towards`** — Hamaina et al. (2012), in *Bridging the Geographic Information Sciences* (Springer). Urban-fabric characterization from building footprints. *Used* (L88) for the facade-continuity concept. ✅

20. **`hautamaki2022modern`** — Hautamäki & Donner (2022), *Geografiska Annaler B* 104(3). Landscape architecture of Finnish "forest suburbs" (retained woodland within suburban expansion). *Used* (L239) to support the Nordic forest-retention → shorter tree-canopy-distance observation. ✅ **Added this round** to substantiate a previously uncited claim.

21. **`herfort2023spatio`** — Herfort et al. (2023), *Nat. Commun.* 14:3985. Uneven global OSM building completeness; ML reliance in some countries. *Used* (L66) for the building-source caveat. ✅

22. **`kropf2009aspects`** — Kropf (2009), *Urban Morphology* 13(2). "Aspects of urban form" — the qualitative morphological traditions. *Used* (L84) to contrast with data-driven classification. ✅

23. **`louf2014typology`** — Louf & Barthelemy (2014), *J. R. Soc. Interface* 11(101). A typology of street patterns from block geometry. *Used* as related work (L58). ✅

24. **`maloutas2003promoting`** — Maloutas (2003), *City* 7(2). Athens social sustainability and the antiparochi/polykatoikia densification. *Used* (L140) for the Greek *antiparochi* account — well supported (a market mechanism that densified Athens/Thessaloniki parcel-by-parcel). ✅

25. **`moudon1997urban`** — Moudon (1997), *Urban Morphology* 1(1). Urban morphology as an emerging interdisciplinary field. *Used* as framing for cross-feature integration (L58). ✅

26. **`oliveira2016urban`** — Oliveira (2016), *Urban Morphology: An Introduction* (Springer). Textbook on the morphological traditions. *Used* as related work / Muratorian context (L58). ✅

27. **`overture2024`** — Overture Maps Foundation data. *Used* (L64) as the source for networks, buildings, and POIs. ✅

28. **`panerai2004urban`** — Panerai, Castex, Depaule & Samuels (2004), *Urban Forms: The Death and Life of the Urban Block* (Architectural Press). The canonical account of the continuous/closed perimeter block (îlot) as the characteristic 19th-c. European fabric and its modernist dissolution. *Used* (L160) for exactly that claim. ✅ **Added this round** to correctly attribute the perimeter-block thesis (previously mis-cited to Caniggia & Maffei).

29. **`simons2021thesis`** — Simons (2021), UCL PhD. 931 GB towns/cities precursor. *Used* (L58) as the direct precursor programme. ✅

30. **`simons2023cityseer`** — Simons (2023), cityseer, *Env. Plan. B* 50(5). *Used* (L60) as the metric-computation engine. ✅

31. **`simons2024soar`** — Simons, Karimi & Zhand, the companion SOAR-EU **data paper** (submitted to *Data in Brief*). *Used* throughout for sources, the POI source-substitution validation, and the SWR-vs-frontage robustness detail. ⚠️ the authors' own companion paper, in submission — not independently verifiable; ensure cross-paper "submitted to" states are correct at submission.

32. **`stanilov2007post`** — Stanilov (ed., 2007), *The Post-Socialist City* (Springer). CEE socialist-era modernist housing estates (slab/tower blocks in open space). *Used* (L142, L160, L261) for the socialist-estate fabric and post-socialist context — well supported. ✅

33. **`starczewski2024green`** — Starczewski et al. (2024), *J. Housing Built Environ.* 39(4). Systematic planned greenery in Polish socialist-era prefab estates. *Used* (L239) for the Polish estate-greenery → shorter canopy-distance observation. ✅ **Added this round** to substantiate a previously uncited claim.

34. **`talen2012city`** — Talen (2012), *City Rules: How Regulations Affect Urban Form* (Island Press). How codes/regulation shape urban form. *Used* (Discussion) for the broad periodisation/regulation argument. ✅

35. **`tsenkova2006urban`** — Tsenkova & Nedović-Budić (eds., 2006), *The Urban Mosaic of Post-Socialist Europe*. Diversity of post-socialist planning/institutional responses. *Used* (L261) for the Poland–Romania divergence. ✅ adequate (a broad volume, not a dedicated Poland–Romania comparison).

36. **`whitehand2001british`** — Whitehand (2001), *Urban Morphology* 5(1). The British/Conzenian town-plan analysis tradition. *Used* (L58) as a foundational tradition. ✅

37. **`vanderhaegen2017mapping`** — Vanderhaegen & Canters (2017), *Landscape Urban Plan.* 167. Mapping urban form/function from building data. *Used* (L88) for the facade-continuity measurement concept. ✅

38. **`yap2023global`** — Yap & Biljecki (2023), *Sci. Data* 10:667. The Urbanity 50-city dataset. *Used* (L54) as a related global comparator. ✅

---

## Changes applied this round (atlas)
- **Re-attributed** the "perimeter block as characteristic European fabric" claim from Caniggia & Maffei to **Panerai et al. (2004)** — `panerai2004urban` added.
- **Re-targeted** `faludi1994rule` to the Dutch plan-led claim and **removed** `buitelaar2011plan` (which argues the opposite of the claim it backed); softened the polder-drainage statement.
- **Added citations** for two previously uncited claims: Polish estate greenery (`starczewski2024green`) and Nordic forest retention (`hautamaki2022modern`).

## Residual notes (judgement calls, not errors)
- `berghauser2019systematic` (atlas L84): the corrected source is a quantitative-typology paper; confirm it supports the specific "clustering is a documented challenge" framing, or add a clustering-methods reference.
- `busquets2005barcelona` (atlas L140): the ensanche wording uses "floor-area ratio" as modern shorthand.
- GHS-UCDB (`ghsl2024`) license label "EC reuse policy" is defensible; the dataset's explicit license is CC BY 4.0 if maximal precision is wanted.
- `simons2024soar` / `soar2026`: self-citations in submission — confirm venue/year and that the Zenodo deposit is public.

## What this audit does not cover
Figure-visual descriptions (e.g. continental scanline gradients) were confirmed against the regenerated macros and, for Plate 9, by inspecting the rendered figure — but the audit does not re-derive every plate. Historical claims rest on the cited literature (verified to exist and, for the load-bearing claims, checked for substantive support).
