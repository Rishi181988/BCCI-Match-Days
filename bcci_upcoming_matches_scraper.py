import requests
import json
import re
import pandas as pd
import time
import os
import argparse
from datetime import datetime
from collections import defaultdict

def safe_strip(value):
    """Handle None values and clean whitespace"""
    return (value or 'N/A').strip()

class BCCIUpcomingMatchesScraper:
    def __init__(self, competitions_json: str | None = None, platforms: list[str] | None = None, genders: list[str] | None = None,
                 competition_ids: list[int] | None = None, competition_name_patterns: list[str] | None = None,
                 month: str | None = None, date_range: tuple[str, str] | None = None, scan_range: tuple[int, int] | None = None):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.bcci.tv/"
        }
        # runtime configuration
        self.platforms = platforms or ["domestic"]
        # genders: values like "men", "women"; default to both
        self.genders = [g.lower() for g in (genders or ["men", "women"])]
        self.competitions_json = competitions_json
        # optional filters
        self.filter_comp_ids = set(competition_ids or [])
        self.filter_comp_name_patterns = [p.lower() for p in (competition_name_patterns or [])]
        self.filter_month = month  # 'YYYY-MM'
        self.filter_date_range = date_range  # (start_date_str, end_date_str)
        self.scan_range = scan_range  # (start_id, end_id)
        # competitions discovered at runtime: {int_id: {"name": str, "gender": str, "platform": str}}
        self.competitions: dict[int, dict] = {}
        self.upcoming_matches = []

    # -----------------------------
    # Competition discovery helpers
    # -----------------------------
    def discover_competitions_from_api(self, platform: str, filter_type: str) -> dict:
        """Discover competitions dynamically from BCCI 'getUpcomingMatches' for a given platform and filter_type.
        Returns a dict mapping competitionId (int) -> {"name": str, "gender": str, "platform": str}.
        """
        url = f"https://scores2.bcci.tv/getUpcomingMatches?platform={platform}&filterType={filter_type}"
        try:
            res = requests.get(url, headers=self.headers, timeout=20)
            res.raise_for_status()
            data = res.json()
            competitions: dict[int, dict] = {}
            # Source 1: explicit list under tournamentsAndSeries
            for item in data.get("tournamentsAndSeries", []) or []:
                cid = item.get("competitionId")
                name = item.get("CompetitionName") or item.get("competitionName")
                if cid is None:
                    continue
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    continue
                competitions[cid_int] = {
                    "name": safe_strip(name),
                    "gender": "men" if filter_type.lower()=="men" else ("women" if filter_type.lower()=="women" else safe_strip(filter_type)),
                    "platform": safe_strip(platform)
                }
            # Source 2: derive from upcomingMatches themselves (fallback)
            for m in data.get("upcomingMatches", []) or []:
                cid = m.get("CompetitionID")
                name = m.get("CompetitionName")
                try:
                    cid_int = int(cid)
                except (TypeError, ValueError):
                    continue
                competitions.setdefault(cid_int, {
                    "name": safe_strip(name),
                    "gender": "men" if filter_type.lower()=="men" else ("women" if filter_type.lower()=="women" else safe_strip(filter_type)),
                    "platform": safe_strip(platform)
                })
            return competitions
        except Exception as e:
            print(f"Failed competition discovery for platform={platform} filter_type={filter_type}: {e}")
            return {}

    def load_competitions_from_json(self, path: str) -> dict:
        """Load competitions mapping from a JSON file.
        Returns {comp_id: {"name": str, "gender": str, "platform": str}}
        Supported formats:
        - Full getUpcomingMatches JSON (with 'tournamentsAndSeries' array)
        - Array of objects with keys {competitionId, CompetitionName, gender?, platform?}
        - Object mapping of {competitionId: CompetitionName}
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            comps: dict[int, dict] = {}
            def default_gender():
                # If the caller requested a single gender, use it; else Unknown
                if len(self.genders) == 1:
                    g = self.genders[0]
                    return "men" if g.lower()=="men" else ("women" if g.lower()=="women" else safe_strip(g))
                return "Unknown"
            def default_platform():
                return self.platforms[0] if self.platforms else "domestic"
            if isinstance(obj, dict):
                # Format A: top-level object mapping
                if any(k for k in obj.keys() if isinstance(k, str) and k.isdigit()):
                    for k, v in obj.items():
                        try:
                            comps[int(k)] = {"name": safe_strip(v), "gender": default_gender(), "platform": default_platform()}
                        except (TypeError, ValueError):
                            continue
                # Format B: getUpcomingMatches response style
                if not comps and 'tournamentsAndSeries' in obj:
                    for item in obj.get('tournamentsAndSeries') or []:
                        try:
                            cid = int(item.get('competitionId'))
                            comps[cid] = {
                                "name": safe_strip(item.get('CompetitionName') or item.get('competitionName')),
                                "gender": safe_strip(item.get('gender') or default_gender()),
                                "platform": safe_strip(item.get('platform') or default_platform()),
                            }
                        except Exception:
                            continue
            if isinstance(obj, list):
                for item in obj:
                    try:
                        cid = int(item.get('competitionId'))
                        comps[cid] = {
                            "name": safe_strip(item.get('CompetitionName') or item.get('competitionName')),
                            "gender": safe_strip(item.get('gender') or default_gender()),
                            "platform": safe_strip(item.get('platform') or default_platform()),
                        }
                    except Exception:
                        continue
            return comps
        except Exception as e:
            print(f"Failed to load competitions from JSON '{path}': {e}")
            return {}

    def build_competitions(self):
        """Build competitions dictionary from JSON config if provided, otherwise via API across selected platforms and genders.
        Optionally scan an explicit ID range to discover more competitions.
        """
        comps: dict[int, dict] = {}
        if self.competitions_json and os.path.exists(self.competitions_json):
            comps = self.load_competitions_from_json(self.competitions_json)
        if not comps:
            merged: dict[int, dict] = {}
            for platform in self.platforms:
                for gender in self.genders:
                    part = self.discover_competitions_from_api(platform, gender)
                    for cid, meta in part.items():
                        # if exists, prefer the first discovered name, but keep gender/platform from the current discovery if not already known
                        if cid not in merged:
                            merged[cid] = meta
                        else:
                            if not merged[cid].get("gender") or merged[cid].get("gender") == "Unknown":
                                merged[cid]["gender"] = meta.get("gender")
                            # keep earliest non-empty platform
                            if not merged[cid].get("platform") or merged[cid].get("platform") == "unknown":
                                merged[cid]["platform"] = meta.get("platform")
            comps = merged
        # Optional scan range to broaden discovery
        if self.scan_range:
            start_id, end_id = self.scan_range
            try:
                scanned = self._scan_competitions_range(int(start_id), int(end_id))
                for cid, meta in scanned.items():
                    comps.setdefault(cid, meta)
                if scanned:
                    print(f"Scan added {len(scanned)} competitions from range {start_id}-{end_id}")
            except Exception as e:
                print(f"Scan range error: {e}")
        self.competitions = comps
        if self.competitions:
            print(f"Discovered {len(self.competitions)} competitions: {sorted(self.competitions.keys())[:10]}{'...' if len(self.competitions)>10 else ''}")
        else:
            print("Warning: No competitions discovered. You can provide a JSON using --competitions-json.")
    # -----------------------------
    # Optional range scanning for competitions
    # -----------------------------
    def _try_fetch_competition_meta(self, comp_id: int) -> dict | None:
        """Attempt to fetch a competition feed and return minimal meta if valid, else None."""
        try:
            url = f"https://scores.bcci.tv/feeds/{comp_id}-matchschedule.js"
            r = requests.get(url, headers=self.headers, timeout=20)
            if r.status_code != 200:
                return None
            json_str = re.sub(r'^MatchSchedule\(|\);?\s*$', '', r.text, flags=re.DOTALL)
            data = json.loads(json_str)
            matches = data.get('Matchsummary') or []
            if not matches:
                return None
            # derive name from first match if available
            name = safe_strip(matches[0].get('CompetitionName')) if matches else f"Competition {comp_id}"
            # infer gender heuristically from name
            upper = name.upper()
            if 'WOMEN' in upper or "WOMEN'S" in upper:
                gender = 'women'
            else:
                gender = 'men'
            platform = (self.platforms[0] if self.platforms else 'domestic')
            return {"name": name, "gender": gender, "platform": platform}
        except Exception:
            return None

    def _scan_competitions_range(self, start_id: int, end_id: int) -> dict:
        """Scan a range of competition IDs and include those that return valid feeds."""
        found: dict[int, dict] = {}
        for cid in range(start_id, end_id + 1):
            if cid in self.competitions or cid in found:
                continue
            meta = self._try_fetch_competition_meta(cid)
            if meta:
                found[cid] = meta
                # be polite
                time.sleep(0.3)
        return found

    # -----------------------------
    # Match filtering helpers
    # -----------------------------
    def _parse_date(self, s: str):
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue
        return None

    def _passes_date_filters(self, date_str: str) -> bool:
        d = self._parse_date(date_str)
        if d is None:
            # If date can't be parsed, do not filter it out
            return True
        if self.filter_month:
            try:
                y, m = self.filter_month.split('-')
                if not (int(y) == d.year and int(m) == d.month):
                    return False
            except Exception:
                pass
        if self.filter_date_range:
            try:
                start_s, end_s = self.filter_date_range
                start_d = datetime.strptime(start_s, "%Y-%m-%d").date()
                end_d = datetime.strptime(end_s, "%Y-%m-%d").date()
                if not (start_d <= d <= end_d):
                    return False
            except Exception:
                pass
        return True

    def should_include_match(self, match_data) -> bool:
        """Return True for ALL matches. Apply only date/month filters if provided."""
        # Always include regardless of status (live, upcoming, finished)
        date_str = match_data.get('MatchDate') or match_data.get('matchDate') or ''
        return self._passes_date_filters(date_str)

    def extract_match_data(self, match, comp_id, comp_name, gender, platform):
        """Extract the required fields from match data"""
        status = match.get('matchStatus') or match.get('MatchStatus') or ''
        # normalize to Title case
        status_norm = safe_strip(status).lower().capitalize() if status else 'N/A'
        return {
            'Platform': 'Domestic' if str(platform).lower()=='domestic' else safe_strip(platform).capitalize(),
            'CompetitionID': comp_id,
            'CompetitionName': safe_strip(match.get('CompetitionName') or comp_name),
            'CompetitionGender': 'Men' if str(gender).lower()=='men' else ('Women' if str(gender).lower()=='women' else safe_strip(gender)),
            'MatchStatus': status_norm,
            'MatchOrder': safe_strip(match.get('MatchOrder')),
            'MatchID': safe_strip(match.get('MatchID')),
            'MatchTypeID': safe_strip(match.get('MatchTypeID')),
            'MatchType': safe_strip(match.get('MatchType')),
            'MatchName': safe_strip(match.get('MatchName')),
            'MatchDate': safe_strip(match.get('MatchDate')),
            'GroundUmpire1': safe_strip(match.get('GroundUmpire1')),
            'GroundUmpire2': safe_strip(match.get('GroundUmpire2')),
            'ThirdUmpire': safe_strip(match.get('ThirdUmpire')),
            'Referee': safe_strip(match.get('Referee')),
            'GroundName': safe_strip(match.get('GroundName'))
        }

    def fetch_competition_data(self, comp_id):
        """Fetch match data for a specific competition"""
        try:
            meta = self.competitions.get(comp_id, {"name": f"Competition {comp_id}", "gender": "Unknown", "platform": "unknown"})
            default_name = meta.get("name", f"Competition {comp_id}")
            gender = meta.get("gender", "Unknown")
            platform = meta.get("platform", "unknown")
            url = f"https://scores.bcci.tv/feeds/{comp_id}-matchschedule.js"

            print(f"Fetching data for {default_name} (ID: {comp_id})...")
            response = requests.get(url, headers=self.headers)

            if response.status_code != 200:
                print(f"Failed to fetch data for competition {comp_id}: HTTP {response.status_code}")
                return 0

            # Clean JSONP response
            json_str = re.sub(r'^MatchSchedule\(|\);?\s*$', '', response.text, flags=re.DOTALL)

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON for competition {comp_id}: {e}")
                return 0

            # Process matches
            matches_processed = 0
            for match in data.get('Matchsummary', []) or []:
                if self.should_include_match(match):
                    # Prefer CompetitionName from the match JSON when available
                    comp_name_from_match = safe_strip(match.get('CompetitionName'))
                    comp_name = comp_name_from_match if comp_name_from_match != 'N/A' else default_name
                    match_data = self.extract_match_data(match, comp_id, comp_name, gender, platform)
                    self.upcoming_matches.append(match_data)
                    matches_processed += 1

            print(f"Found {matches_processed} matches for {default_name}")
            return matches_processed

        except Exception as e:
            print(f"Error processing competition {comp_id}: {str(e)}")
            return 0

    def scrape_all_competitions(self):
        """Scrape matches from all discovered competitions with optional filters"""
        print("Starting BCCI matches scraping...\n")
        total_matches = 0

        # Build the competitions map dynamically if not already provided
        if not self.competitions:
            self.build_competitions()

        # Apply competition ID/name filters
        comp_items = list(self.competitions.items())
        if self.filter_comp_ids:
            comp_items = [(cid, meta) for cid, meta in comp_items if cid in self.filter_comp_ids]
        if self.filter_comp_name_patterns:
            patterns = self.filter_comp_name_patterns
            comp_items = [
                (cid, meta) for cid, meta in comp_items
                if any(p in (meta.get('name','').lower()) for p in patterns)
            ]

        for comp_id, _meta in comp_items:
            matches_count = self.fetch_competition_data(comp_id)
            total_matches += matches_count

            # Add delay to be respectful to the server
            time.sleep(1.5)

        print(f"\nScraping complete! Found {total_matches} total matches.")
        return total_matches

    def export_to_excel(self, filename="bcci_upcoming_matches.xlsx"):
        """Export the scraped data to Excel"""
        if not self.upcoming_matches:
            print("No upcoming matches found to export.")
            return

        # Create DataFrame
        df = pd.DataFrame(self.upcoming_matches)

        # Reorder columns to: Platform, CompetitionGender, CompetitionID, CompetitionName, MatchStatus, then the rest of the 13 fields
        desired_order = [
            'Platform', 'CompetitionGender', 'CompetitionID', 'CompetitionName', 'MatchStatus',
            'MatchOrder', 'MatchID', 'MatchTypeID', 'MatchType', 'MatchName', 'MatchDate',
            'GroundUmpire1', 'GroundUmpire2', 'ThirdUmpire', 'Referee', 'GroundName'
        ]
        cols = [c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]
        df = df[cols]

        # Sort by Platform, CompetitionGender, CompetitionID and MatchDate
        sort_cols = [c for c in ['Platform', 'CompetitionGender', 'CompetitionID', 'MatchDate'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        # Export to Excel with formatting
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Upcoming Matches', index=False)

            # Get the workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Upcoming Matches']

            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#FFFF00',
                'valign': 'vcenter',
                'align': 'center',
                'text_wrap': True,
                'border': 1
            })

            # Data format
            data_format = workbook.add_format({
                'valign': 'vcenter',
                'align': 'center',
                'border': 1
            })

            # Apply header formatting
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Set column widths and apply data formatting
            column_widths = {
                'Platform': 12,
                'CompetitionGender': 12,
                'CompetitionID': 12,
                'CompetitionName': 28,
                'MatchStatus': 12,
                'MatchOrder': 12,
                'MatchID': 12,
                'MatchTypeID': 12,
                'MatchType': 22,
                'MatchName': 34,
                'MatchDate': 14,
                'GroundUmpire1': 22,
                'GroundUmpire2': 22,
                'ThirdUmpire': 20,
                'Referee': 20,
                'GroundName': 28
            }

            for col_num, column in enumerate(df.columns):
                width = column_widths.get(column, 15)
                worksheet.set_column(col_num, col_num, width, data_format)

            # Apply borders to all data cells
            max_row = len(df)
            max_col = len(df.columns) - 1
            worksheet.conditional_format(1, 0, max_row, max_col, {
                'type': 'no_blanks',
                'format': data_format
            })

        print(f"Data exported to {filename}")
        print(f"Total upcoming matches exported: {len(df)}")

        # Display sample data
        print(f"\nSample of exported data:")
        print(df[['CompetitionName', 'MatchName', 'MatchDate', 'GroundName']].head(3).to_string(index=False))

    def print_summary(self):
        """Print a summary of the scraped data"""
        if not self.upcoming_matches:
            print("No upcoming matches found.")
            return

        print(f"\n=== UPCOMING MATCHES SUMMARY ===")
        print(f"Total upcoming matches: {len(self.upcoming_matches)}")

        # Group by competition
        competition_counts = defaultdict(int)
        for match in self.upcoming_matches:
            competition_counts[match['CompetitionName']] += 1

        print("\nMatches by competition:")
        for comp_name, count in sorted(competition_counts.items()):
            print(f"  {comp_name}: {count} matches")

    def save_raw_data(self, filename="bcci_raw_data.json"):
        """Save raw match data as JSON for debugging purposes"""
        if not self.upcoming_matches:
            print("No data to save.")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.upcoming_matches, f, indent=2, ensure_ascii=False)

        print(f"Raw data saved to {filename}")

def main():
    """Main function to run the scraper with optional JSON-based configuration"""
    parser = argparse.ArgumentParser(description="BCCI Matches Scraper (all statuses, men & women)")
    parser.add_argument("--competitions-json", dest="competitions_json", help="Path to JSON file with competitions metadata (optional)")
    parser.add_argument("--platforms", default="domestic", help="Comma-separated platforms to include (e.g., domestic,international)")
    parser.add_argument("--genders", default="both", help="men | women | both (default: both)")
    parser.add_argument("--competition-ids", default=None, help="Comma-separated competition IDs to include (e.g., 316,317)")
    parser.add_argument("--competition-names", default=None, help="Comma-separated name substrings to include (case-insensitive, e.g., ranji,irani)")
    parser.add_argument("--month", default=None, help="Filter matches to a specific month (YYYY-MM)")
    parser.add_argument("--date-range", default=None, help="Filter matches within a date range (YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument("--scan-range", default=None, help="Scan a range of competition IDs (e.g., 300-400) to discover more comps")
    parser.add_argument("--output", default="bcci_upcoming_matches.xlsx", help="Output Excel filename")
    # Backward-compatible flags
    parser.add_argument("--platform", default=None, help="Deprecated: single platform; use --platforms")
    parser.add_argument("--filter-type", default=None, help="Deprecated: men/women; use --genders")
    args = parser.parse_args()

    # derive platforms list
    platforms = []
    if args.platforms:
        platforms = [p.strip() for p in args.platforms.split(',') if p.strip()]
    elif args.platform:
        platforms = [args.platform.strip()]
    else:
        platforms = ["domestic"]

    # derive genders list
    g = (args.genders or "both").strip().lower()
    if g == "both":
        genders = ["men", "women"]
    elif g in ("men", "women"):
        genders = [g]
    elif args.filter_type in ("men", "women"):
        genders = [args.filter_type]
    else:
        genders = ["men", "women"]

    # competition ids
    comp_ids = None
    if args.competition_ids:
        try:
            comp_ids = [int(x) for x in args.competition_ids.split(',') if x.strip()]
        except Exception:
            comp_ids = None

    # competition names
    comp_names = None
    if args.competition_names:
        comp_names = [x.strip() for x in args.competition_names.split(',') if x.strip()]

    # date filters
    date_range = None
    if args.date_range and ',' in args.date_range:
        parts = [p.strip() for p in args.date_range.split(',')]
        if len(parts) == 2:
            date_range = (parts[0], parts[1])

    month = args.month

    # scan range
    scan_range = None
    if args.scan_range and '-' in args.scan_range:
        try:
            a, b = args.scan_range.split('-')
            scan_range = (int(a.strip()), int(b.strip()))
        except Exception:
            scan_range = None

    scraper = BCCIUpcomingMatchesScraper(
        competitions_json=args.competitions_json,
        platforms=platforms,
        genders=genders,
        competition_ids=comp_ids,
        competition_name_patterns=comp_names,
        month=month,
        date_range=date_range,
        scan_range=scan_range,
    )

    # Scrape all competitions
    scraper.scrape_all_competitions()

    # Print summary
    scraper.print_summary()

    # Export to Excel
    scraper.export_to_excel(args.output)

    # Optionally save raw JSON data
    scraper.save_raw_data()

if __name__ == "__main__":
    main()
