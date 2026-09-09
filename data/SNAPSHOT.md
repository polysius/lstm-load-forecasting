# Data Snapshot

**Source:** https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
**Download date:** 2026-09-09
**Target client:** MT_001

## Validated properties

- **Rows:** 26305
- **Date range:** 2012-01-01 00:00:00+00:00 to 2015-01-01 00:00:00+00:00
- **Frequency:** 1h
- **Timezone:** Europe/Lisbon

## Missing data

- **NaN values:** 0 (0.0%)
- **Gaps:** 0
- **Total gap hours:** 0
- **Max gap hours:** 0

## Preprocessing

- Resampled to 1h
- Interpolation: linear, max gap 3h
- Gaps > 3h left as NaN
- 2011 dropped entirely (MT_001 has zero readings for all of 2011; meter offline)
- 40,743 zero readings in raw 15-min data replaced with NaN before resampling
