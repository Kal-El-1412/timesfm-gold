"""
Scheduled macro-event calendar for gold event-window research (Phase 3, A1).

We do NOT have consensus/surprise data, so this is the cheaper proxy: flag the
DATES of high-impact scheduled US releases and test whether gold's behaviour
concentrates around them.

Two sources, by reliability:
  * NFP  - US non-farm payrolls. Released the FIRST FRIDAY of each month
           (rule-based, derived here -> reliable).
  * FOMC - rate decisions. 8 scheduled meetings/year; announcement dates are
           hard-coded below from the public Fed schedule.

  >>> VERIFY the FOMC list against federalreserve.gov before relying on it for
  >>> anything beyond an exploratory signal scan. A handful of dates may be off
  >>> by a day; that is fine for a window test (we widen with +/- a day) but not
  >>> for event-study precision. Swap in a real calendar feed for A2.

You can also override everything by passing a CSV of dates to load_event_dates.
"""

import pandas as pd

# Announcement dates (mostly the Wednesday of the 2-day meeting). Best-effort
# from the public FOMC schedule 2016-2026 -- VERIFY before production use.
FOMC_DATES = [
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]


def nfp_dates(start, end):
    """First Friday of each month in [start, end] -> NFP release dates."""
    months = pd.date_range(pd.Timestamp(start).normalize().replace(day=1),
                           pd.Timestamp(end).normalize(), freq="MS")
    out = []
    for m in months:
        fridays = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")
        if len(fridays):
            out.append(fridays[0].normalize())
    return pd.DatetimeIndex(out)


def fomc_dates():
    return pd.DatetimeIndex(pd.to_datetime(FOMC_DATES)).normalize()


def build_event_flags(index, window=1, csv=None):
    """
    Given a DatetimeIndex, return a DataFrame of event flags aligned to it.

    window: a row is "in the event window" if it is within +/- `window`
            trading rows of an event date (captures pre-positioning and the
            next-day reaction, and absorbs +/-1 day calendar imprecision).
    """
    index = pd.DatetimeIndex(index).normalize()
    if csv:
        ev = pd.read_csv(csv, parse_dates=["date"])
        nfp = ev.loc[ev["event"].str.upper() == "NFP", "date"]
        fomc = ev.loc[ev["event"].str.upper() == "FOMC", "date"]
        nfp = pd.DatetimeIndex(nfp).normalize()
        fomc = pd.DatetimeIndex(fomc).normalize()
    else:
        nfp = nfp_dates(index.min(), index.max())
        fomc = fomc_dates()

    def nearest_flag(dates):
        # map each event date to the nearest trading row, then expand by window
        pos = index.get_indexer(dates, method="nearest")
        pos = pos[pos >= 0]
        flag = pd.Series(0, index=index)
        for p in pos:
            lo, hi = max(0, p - window), min(len(index) - 1, p + window)
            flag.iloc[lo:hi + 1] = 1
        return flag

    df = pd.DataFrame(index=index)
    df["is_nfp"] = nearest_flag(nfp).values
    df["is_fomc"] = nearest_flag(fomc).values
    df["is_event"] = ((df["is_nfp"] == 1) | (df["is_fomc"] == 1)).astype(int)
    return df


if __name__ == "__main__":
    idx = pd.date_range("2016-08-26", "2026-06-11", freq="B")
    flags = build_event_flags(idx, window=1)
    print("Business days:", len(idx))
    print("NFP-window days :", int(flags["is_nfp"].sum()))
    print("FOMC-window days:", int(flags["is_fomc"].sum()))
    print("Any-event days  :", int(flags["is_event"].sum()))
