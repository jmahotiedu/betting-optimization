# EV Schema Probe

## Columns in transactions.csv

bet_id, sportsbook, type, status, odds, closing_line, ev, amount, profit, time_placed, time_settled, time_placed_iso, time_settled_iso, bet_info, tags, sports, leagues

## 25-row sample of EV-related fields

| ev | odds | closing_line | amount | profit | bet_info | sportsbook | time_placed_iso | time_placed | time_settled | type | status | sports | leagues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0305 | 2.27 | 2.2 | 24.22 | 30.69 | over 34.5 Lauri Markkanen Total Points, Rebounds & Assists Utah Jazz at San A... | ProphetX | 2025-12-27T22:55:44.000Z | 12/27/2025 22:55:44 GMT | 12/28/2025 04:00:00 GMT | straight | SETTLED_WIN | Basketball | NBA |
| 0.1111 | 2.22 | 2.0 | 16.9965 | -17.0 | Under 23.5 Anton Forsberg SAVES Anaheim Ducks @ Los Angeles Kings | Novig | 2025-12-28T00:03:28.502Z | 12/28/2025 00:03:28 GMT | 12/28/2025 02:00:00 GMT | straight | SETTLED_LOSS | Ice Hockey | NHL |
| 0.0742 | 2.32 | 2.16 | 4.54274 | 6.0 | HOU -2.5 SPREAD 1H Cleveland Cavaliers @ Houston Rockets | Novig | 2025-12-28T00:46:13.660Z | 12/28/2025 00:46:13 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_WIN | Basketball | NBA |
| 0.1098 | 2.2 | 1.98 | 0.0091 | -0.01 | Over 10.5 AJ Green POINTS Milwaukee Bucks @ Chicago Bulls | Novig | 2025-12-27T23:56:26.567Z | 12/27/2025 23:56:26 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.0535 | 2.21 | 2.1 | 21.696 | -21.7 | Over 12.5 Matas Buzelis POINTS Milwaukee Bucks @ Chicago Bulls | Novig | 2025-12-27T23:29:56.240Z | 12/27/2025 23:29:56 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.0525 | 2.25 | 2.14 | 22.2 | 27.8 | Under 21.5 Trae Young POINTS New York Knicks @ Atlanta Hawks | Novig | 2025-12-27T23:38:10.951Z | 12/27/2025 23:38:10 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_WIN | Basketball | NBA |
| 0.0582 | 2.38 | 2.25 | 33.4992 | 46.26 | Over 15.5 Luke Kornet POINTS REBOUNDS ASSISTS Utah Jazz @ San Antonio Spurs | Novig | 2025-12-27T23:32:01.201Z | 12/27/2025 23:32:01 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_WIN | Basketball | NBA |
| 0.0356 | 2.25 | 2.17 | 20.03835 | -20.04 | Under 1.5 Tari Eason THREE POINTERS MADE Cleveland Cavaliers @ Houston Rockets | Novig | 2025-12-27T23:40:47.404Z | 12/27/2025 23:40:47 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| -0.1247 | 2.19 | 2.5 | 17.19691 | -17.2 | Over 20.5 Kyle Kuzma POINTS REBOUNDS ASSISTS Milwaukee Bucks @ Chicago Bulls | Novig | 2025-12-27T23:46:24.383Z | 12/27/2025 23:46:24 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.0535 | 2.21 | 2.1 | 22.6 | -22.6 | Over 12.5 Matas Buzelis POINTS Milwaukee Bucks @ Chicago Bulls | Novig | 2025-12-27T23:35:45.016Z | 12/27/2025 23:35:45 GMT | 12/28/2025 01:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.0659 | 2.1 | 1.97 | 21.09632 | 23.22 | Over 24.5 Tyler Huntley RUSHING YARDS Baltimore Ravens @ Green Bay Packers | Novig | 2025-12-28T00:10:43.824Z | 12/28/2025 00:10:43 GMT | 12/28/2025 01:00:00 GMT | straight | SETTLED_WIN | American Football | NFL |
| 0.0799 | 2.16 | 2.0 | 17.19582 | -17.2 | Over 4.5 Derik Queen ASSISTS Phoenix Suns @ New Orleans Pelicans | Novig | 2025-12-27T23:36:16.721Z | 12/27/2025 23:36:16 GMT | 12/28/2025 00:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| -0.153 | 2.16 | 2.55 | 24.79828 | -24.8 | Over 13.5 Mark Williams POINTS Phoenix Suns @ New Orleans Pelicans | Novig | 2025-12-27T23:38:54.160Z | 12/27/2025 23:38:54 GMT | 12/28/2025 00:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.1895 | 1.87 | 1.57 | 29.99745 | -30.0 | Under 4.5 Nique Clifford REBOUNDS Dallas Mavericks @ Sacramento Kings | Novig | 2025-12-27T18:42:46.591Z | 12/27/2025 18:42:46 GMT | 12/27/2025 22:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.0164 | 2.24 | 2.2 | 16.1 | -16.1 | Portland Winner LAC at POR | Kalshi | 2025-12-26T18:31:22.621Z | 12/26/2025 18:31:22 GMT | 12/27/2025 05:59:31 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.1174 | 2.09 | 1.87 | 4.55 | 4.96 | under 26.5 Derik Queen Total Points, Rebounds & Assists Phoenix Suns at New O... | ProphetX | 2025-12-26T20:29:42.000Z | 12/26/2025 20:29:42 GMT | 12/27/2025 03:55:30 GMT | straight | SETTLED_WIN | Basketball | NBA |
| 0.0339 | 2.15 | 2.08 | 20.9994 | 24.16 | Under 6.5 John Collins REBOUNDS Los Angeles Clippers @ Portland Trail Blazers | Novig | 2025-12-26T21:16:12.205Z | 12/26/2025 21:16:12 GMT | 12/27/2025 03:05:00 GMT | straight | SETTLED_WIN | Basketball | NBA |
| -0.1241 | 2.35 | 2.68 | 21.39798 | 28.83 | Under 20.5 John Collins POINTS REBOUNDS ASSISTS Los Angeles Clippers @ Portla... | Novig | 2025-12-26T18:06:02.789Z | 12/26/2025 18:06:02 GMT | 12/27/2025 03:05:00 GMT | straight | SETTLED_WIN | Basketball | NBA |
| 0.0331 | 2.25 | 2.18 | 21.39636 | -21.4 | Under 10.5 Duncan Robinson POINTS Detroit Pistons @ Utah Jazz | Novig | 2025-12-26T16:13:15.632Z | 12/26/2025 16:13:15 GMT | 12/27/2025 02:35:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.0721 | 2.12 | 1.98 | 19.59831 | -19.6 | Under 15.5 Duncan Robinson POINTS REBOUNDS ASSISTS Detroit Pistons @ Utah Jazz | Novig | 2025-12-26T20:24:33.547Z | 12/26/2025 20:24:33 GMT | 12/27/2025 02:35:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.0817 | 2.33 | 2.15 | 15.4972 | -15.5 | Under 8.5 Jusuf Nurkic REBOUNDS Detroit Pistons @ Utah Jazz | Novig | 2025-12-26T19:01:10.064Z | 12/26/2025 19:01:10 GMT | 12/27/2025 02:35:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| -0.0584 | 2.12 | 2.25 | 19.09712 | -19.1 | Under 12.5 Mark Williams POINTS Phoenix Suns @ New Orleans Pelicans | Novig | 2025-12-26T16:15:07.117Z | 12/26/2025 16:15:07 GMT | 12/27/2025 01:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| -0.078 | 2.03 | 2.2 | 26.59735 | -26.6 | Under 5.5 Collin Gillespie ASSISTS Phoenix Suns @ New Orleans Pelicans | Novig | 2025-12-26T18:46:11.857Z | 12/26/2025 18:46:11 GMT | 12/27/2025 01:05:00 GMT | straight | SETTLED_LOSS | Basketball | NBA |
| 0.1716 | 2.32 | 1.98 | 20.00702 | 26.41 | Under 13.5 Derik Queen POINTS Phoenix Suns @ New Orleans Pelicans | Novig | 2025-12-26T18:03:52.774Z | 12/26/2025 18:03:52 GMT | 12/27/2025 01:05:00 GMT | straight | SETTLED_WIN | Basketball | NBA |
| 0.0589 | 2.14 | 2.02 | 19.99536 | 22.77 | Under 25.5 Derik Queen POINTS REBOUNDS ASSISTS Phoenix Suns @ New Orleans Pel... | Novig | 2025-12-26T20:32:51.655Z | 12/26/2025 20:32:51 GMT | 12/27/2025 01:05:00 GMT | straight | SETTLED_WIN | Basketball | NBA |