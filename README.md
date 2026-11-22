# Not Every Peak Is a Beat: R-Peak Detection Errors in Single-Lead Holter ECG

DAMSS 2026 poster. Nika Bukolovaitė, Zigmantas Kęstutis Juškevičius, Žygimantas Abramikas, Viktor Medvedev, Jolita Bernatavičienė.

## Why it happens or what helps

Supplement to Table 1 of the poster.

- A missed beat is a beat of the vote with no mark within 150 ms, and an extra mark is a mark with no such beat.
- The vote is the majority vote of NeuroKit, Pan–Tompkins, Hamilton and XQRS, where at least 3 of 4 must mark a beat within 150 ms.
- Fix: the signal is inverted if most QRS complexes are negative, and each mark is moved to the largest deflection within 50 ms.

| Problem (example)                                                                          | Why it happens or what helps                                                                                   |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **R-peak on the T wave**<br>Recording 1704502 (Fig. 1a, 2a)                                | **Helps:** a narrow 5–20 Hz band, which weakens T waves.                                                       |
| **Extra marks between beats**<br>Recordings 1715213, 1652786, 1716089 (Fig. 1b, c, f)      | **Helps:** another detector. XQRS with fix does not mark these small peaks.                                    |
| **Missed beats**<br>Recording 1633509: 7 unusual beats (Fig. 1d)                           | **Helps:** another detector. All four voting detectors find these beats.                                       |
| **QRS pointing down**<br>4 of 16 recordings, e.g. 1640106 (Fig. 1e, 2b)                    | **Helps:** detectors that work on the squared or absolute signal, so the direction of the QRS does not matter. |
| **Inverted T wave, changed QRS**<br>Recordings 1640159, 1640137 (left bundle branch block) | **Probably helps:** handling the QRS direction and a band wide enough for a wide QRS.                          |
| **Mark away from the QRS peak**<br>Pan–Tompkins and Hamilton                               | **Helps:** the fix, which moves each mark to the largest deflection within 50 ms.                              |
