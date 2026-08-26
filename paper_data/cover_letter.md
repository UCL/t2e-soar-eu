# Cover letter — Scientific Data

Dear Editors,

We submit the Data Descriptor "SOAR-EU: A Scalable, Open, Automatable, and Reproducible European Urban Dataset" for consideration in Scientific Data.

SOAR-EU provides pre-computed pedestrian-scale metrics for 626 urban centres across 28 European countries. Each of over 12 million natural street segments carries over 100 metrics spanning network centrality, land-use and infrastructure accessibility, mixed-use diversity, building and block morphology, green-space and tree-canopy proximity, and interpolated census demographics. All metrics are computed with identical methods, parameters, and open sources (Overture Maps, Copernicus, Eurostat, GHS-UCDB), and most are provided at multiple network-distance thresholds (200–9,600 m) so that users can match the measurement scale to their question.

Technical validation goes beyond descriptive checks. A source-substitution analysis recomputes POI-derived metrics for the six registry-comparable categories against official national registries (SIRENE in France, BAG in the Netherlands) across 116 reference cities, holding the street network constant, and translates the resulting agreement levels into task- and scale-specific usage guidance. Building-footprint provenance effects are quantified per metric, and a per-column completeness report ships with the deposit.

The dataset is deposited at Zenodo (DOI 10.5281/zenodo.18961227) under ODbL 1.0; the full processing pipeline is open source (AGPL-3.0, github.com/UCL/t2e-soar-eu) and regenerates the dataset end to end in approximately one day, so the resource can be refreshed as upstream sources release new versions. A companion research article applying the dataset to a morphological classification of European cities has been submitted to Computers, Environment and Urban Systems.

The manuscript is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. The work was funded by the European Union's Horizon Europe programme (Grant Agreement No. 101078890) and UKRI Horizon Europe guarantee grants 10052856 and 10050784.

Suggested reviewers: [to be completed by the authors]

Yours sincerely,
Gareth Simons (corresponding author), Kayvan Karimi, Sepehr Zhand
Space Syntax Laboratory, UCL Bartlett School of Architecture
