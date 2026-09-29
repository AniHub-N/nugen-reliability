# Sources

Both documents are public government publications, downloaded on 2026-09-29.

| File in `data/raw/` | What it is | URL | sha256 |
|---|---|---|---|
| `rera_act_2016.pdf` | The Real Estate (Regulation and Development) Act, 2016 (No. 16 of 2016), Gazette of India Extraordinary, Part II Section 1, 26 March 2016 | https://rera.mohua.gov.in/images/pdf_folder/Real_Estate_Act_2016(2).pdf (linked from https://rera.mohua.gov.in/real-estate-regulation-and-development-act-2016.html) | `b825a485348ad58cdf7073d029d5c15bb0315dca0f2867ff5f0f30145e89781f` |
| `telangana_rera_rules_2017.pdf` | Telangana State Real Estate (Regulation and Development) Rules, 2017, G.O.Ms.No.202, MA&UD (M1) Dept, 31-07-2017 | https://rerait.telangana.gov.in/PDF/Telangana%20State%20Real%20Estate%20_Regulation%20and%20Development_%20Rules-2017.pdf | `155ef9da64d499c449cc76e98273e7f6a2242b5b73e107288faae4306b6e7e14` |

Notes

- The MoHUA RERA portal was serving an expired TLS certificate on the download date, so the Act was fetched with certificate verification off (`curl -k`). The file is the Gazette print, 37 pages; its text matches the India Code version section by section (all 92 sections present).
- The mohua.gov.in and indiacode.nic.in links that search engines return for the Act were dead (404 / CDN error) on that date.
- The Telangana file is the rules as originally notified in 2017. Later amendments (for example G.O.Ms.No.60 of 2025) and the TG-RERA General Regulations 2023 are not in the corpus, so questions about them count as out of scope.

`data/corpus/*.txt` is generated from these PDFs by `python scripts/build_corpus.py`. Section numbers are kept as printed. In the Act, the margin headings are put back in front of their sections as `[Section heading: ...]`.
