import requests
import json
import re
import pandas as pd
import time
import os
import argparse
from datetime import datetime
from collections import defaultdict
from urllib.parse import quote

import sys
from io import BytesIO
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except Exception:
    _STREAMLIT_AVAILABLE = False

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
    # Slug helpers
    # -----------------------------
    @staticmethod
    def _slug_space(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
        return re.sub(r"\s+", " ", s)

    @staticmethod
    def _slug_hyphen(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip('-')
        return re.sub(r"-+", "-", s)


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
                if cid is None or not name:
                    continue
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
    def discover_competitions_from_competition_js(self) -> dict:
        """Fetch all competitions from competition.js JSONP and build mapping.
        Returns {comp_id: {name, gender, platform, series_slug, page_slug, category, teamType}}
        """
        url = "https://scores.bcci.tv/feeds/competition.js?get=params&callback=oncomptetion&_=" + str(int(time.time()*1000))
        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            # Strip JSONP wrapper oncomptetion(...)
            body = r.text.strip()
            body = re.sub(r"^oncomptetion\(", "", body)
            body = re.sub(r"\)\s*;?\s*$", "", body)
            data = json.loads(body)
            comps = {}
            for item in data.get('competition', []) or []:
                try:
                    cid = int(item.get('CompetitionID'))
                except Exception:
                    continue
                name = safe_strip(item.get('CompetitionName'))
                team_type = safe_strip(item.get('TeamType'))  # e.g., Mens Senior, Womens Senior
                gender = 'women' if team_type.lower().startswith('wom') else 'men'
                space_slug = self._slug_space(name)
                hyphen_slug = self._slug_hyphen(name)
                comps[cid] = {
                    'name': name,
                    'gender': gender,
                    'platform': 'domestic',
                    'series_slug': space_slug,
                    'page_slug': hyphen_slug,
                    'category': safe_strip(item.get('Category')),
                    'teamType': team_type,
                }
            return comps
        except Exception as e:
            print(f"competition.js discovery failed: {e}")
            return {}

    def fetch_competition_page(self, comp_id: int, slug: str) -> dict:
        """Fetch competition page for optional enrichment. Non-fatal if fails.
        Returns a minimal dict of extra metadata if anything obvious is found.
        """
        url = f"https://www.bcci.tv/domestic/{comp_id}/{slug}"
        try:
            res = requests.get(url, headers=self.headers, timeout=30)
            if res.status_code != 200:
                return {}
            html = res.text
            # Light extraction: look for a canonical title meta or h1
            title = None
            m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
            if m:
                title = safe_strip(re.sub(r"<[^>]+>", " ", m.group(1)))
            if title and title.upper() != 'N/A':
                return {'page_title': title}
            return {}
        except Exception:
            return {}

    def fetch_match_center_details(self, series_slug: str, match_id, series_id):
        """Fetch detailed match-center JSON. Non-fatal if fails.
        series_slug must be URL-encoded; match_id/series_id can be str or int.
        """
        try:
            slug_enc = quote(series_slug or "", safe="")
            url = f"https://scores2.bcci.tv/getDomesticMatchCenterDetails?seriesSlug={slug_enc}&matchID={match_id}&SERIES_ID={series_id}"
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"match center fetch failed for series_slug={series_slug}, match_id={match_id}, series_id={series_id}: {e}")
            return None



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
            # Prefer the comprehensive competition.js discovery
            comp_js = self.discover_competitions_from_competition_js()
            if comp_js:
                comps = {
                    cid: {
                        'name': meta.get('name'),
                        'gender': meta.get('gender', 'Unknown'),
                        'platform': meta.get('platform', 'domestic'),
                        'series_slug': meta.get('series_slug'),
                        'page_slug': meta.get('page_slug'),
                        'category': meta.get('category'),
                        'teamType': meta.get('teamType'),
                    }
                    for cid, meta in comp_js.items()
                }
                # Optional enrichment via competition page
                for cid, meta in list(comps.items()):
                    slug = meta.get('page_slug') or self._slug_hyphen(meta.get('name', ''))
                    extra = self.fetch_competition_page(cid, slug)
                    if extra.get('page_title'):
                        comps[cid]['name'] = extra['page_title']
            else:
                # Fallback to older discovery via getUpcomingMatches across selected platforms/genders
                merged: dict[int, dict] = {}
                for platform in self.platforms:
                    for gender in self.genders:
                        part = self.discover_competitions_from_api(platform, gender)
                        for cid, meta in part.items():
                            # if exists, prefer the first discovered name, but keep gender/platform from the current discovery if not already known
                            if cid not in merged:
                                merged[cid] = meta
                            else:
                                if not merged[cid].get('gender') or merged[cid].get('gender') == 'Unknown':
                                    merged[cid]['gender'] = meta.get('gender')
                                # keep earliest non-empty platform
                                if not merged[cid].get('platform') or merged[cid].get('platform') == 'unknown':
                                    merged[cid]['platform'] = meta.get('platform')
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
            name_raw = matches[0].get('CompetitionName') if matches else None
            name = safe_strip(name_raw) if name_raw else f"Competition {comp_id}"
            if name == 'N/A':
                name = f"Competition {comp_id}"
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

    def _compute_number_of_days(self, match_type: str):
        s = (match_type or '').strip().lower()
        if not s:
            return ''
        # direct mappings
        if s in ('t20', 't20 d/n', 't20 d\n'):  # guard unusual variants
            return 0.5
        if s in ('one day', 'one day d/n', 'one day d\n'):
            return 1
        # multi-day pattern: Multi Day (3 Days)
        m = re.match(r"multi\s*day\s*\((\d+)\s*days?\)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return ''
        return ''

    def extract_match_data(self, match, comp_id, comp_name, gender, platform):
        """Extract the required fields from match data"""
        status = match.get('matchStatus') or match.get('MatchStatus') or ''
        # normalize to Title case
        status_norm = safe_strip(status).lower().capitalize() if status else 'N/A'
        match_type = safe_strip(match.get('MatchType'))
        num_days = self._compute_number_of_days(match_type)
        return {
            'Platform': 'Domestic' if str(platform).lower()=='domestic' else safe_strip(platform).capitalize(),
            'CompetitionID': comp_id,
            'CompetitionName': safe_strip(match.get('CompetitionName') or comp_name),
            'CompetitionGender': 'Men' if str(gender).lower()=='men' else ('Women' if str(gender).lower()=='women' else safe_strip(gender)),
            'MatchStatus': status_norm,
            'MatchOrder': safe_strip(match.get('MatchOrder')),
            'MatchID': safe_strip(match.get('MatchID')),
            'MatchTypeID': safe_strip(match.get('MatchTypeID')),
            'MatchType': match_type,
            'Number of Days': num_days,
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

            # Derive a better competition name if feed contains it anywhere
            names = [safe_strip(m.get('CompetitionName')) for m in (data.get('Matchsummary', []) or [])]
            non_na_names = [n for n in names if n and n != 'N/A']
            if non_na_names:
                default_name = non_na_names[0]
                # update cache
                self.competitions.setdefault(comp_id, {}).update({"name": default_name})

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

        for comp_id, _ in comp_items:
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

        # Reorder columns to: Platform, CompetitionGender, CompetitionID, CompetitionName, MatchStatus, then the rest fields (+ Number of Days)
        desired_order = [
            'Platform', 'CompetitionGender', 'CompetitionID', 'CompetitionName', 'MatchStatus',
            'MatchOrder', 'MatchID', 'MatchTypeID', 'MatchType', 'Number of Days', 'MatchName', 'MatchDate',
            'GroundUmpire1', 'GroundUmpire2', 'ThirdUmpire', 'Referee', 'GroundName'
        ]
        cols = [c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]
        df = df[cols]

        # Sort by Platform, CompetitionGender, CompetitionID and MatchDate
        sort_cols = [c for c in ['Platform', 'CompetitionGender', 'CompetitionID', 'MatchDate'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        # Prepare pivot dataframes before writing
        def _is_valid_name(x: str) -> bool:
            if x is None:
                return False
            s = str(x).strip()
            return bool(s) and s.upper() != 'N/A'
        df_days = df.copy()
        df_days['Number of Days'] = pd.to_numeric(df_days.get('Number of Days', 0), errors='coerce').fillna(0.0)
        cols_base = ['CompetitionName', 'Number of Days']
        comps_sorted = sorted([c for c in df_days['CompetitionName'].dropna().unique().tolist() if c and c != 'N/A'])
        # Umpires pivot
        u1 = df_days[cols_base + ['GroundUmpire1']].rename(columns={'GroundUmpire1': 'Umpire Name', 'CompetitionName': 'Tournament Name'})
        u2 = df_days[cols_base + ['GroundUmpire2']].rename(columns={'GroundUmpire2': 'Umpire Name', 'CompetitionName': 'Tournament Name'})
        umpires = pd.concat([u1, u2], ignore_index=True)
        umpires = umpires[umpires['Umpire Name'].apply(_is_valid_name)]
        ump_pivot = (umpires
                     .pivot_table(index='Umpire Name', columns='Tournament Name', values='Number of Days', aggfunc='sum', fill_value=0.0)
                     .reset_index())
        ordered_cols = ['Umpire Name'] + comps_sorted
        for c in ordered_cols:
            if c not in ump_pivot.columns:
                ump_pivot[c] = 0.0
        missing_cols = [c for c in ump_pivot.columns if c not in ordered_cols and c != 'Umpire Name']
        ordered_cols = ['Umpire Name'] + comps_sorted + missing_cols
        ump_pivot = ump_pivot[ordered_cols]
        ump_pivot['Total'] = ump_pivot[[c for c in ump_pivot.columns if c not in ('Umpire Name', 'Total')]].sum(axis=1)
        ump_pivot = ump_pivot.sort_values(['Total', 'Umpire Name'], ascending=[False, True])
        # Referee pivot
        ref = df_days[['CompetitionName', 'Number of Days', 'Referee']].rename(
            columns={'Referee': 'Referee Name', 'CompetitionName': 'Tournament Name'})
        ref = ref[ref['Referee Name'].apply(_is_valid_name)]
        ref_pivot = (ref
                     .pivot_table(index='Referee Name', columns='Tournament Name', values='Number of Days', aggfunc='sum', fill_value=0.0)
                     .reset_index())
        ordered_cols_r = ['Referee Name'] + comps_sorted
        for c in ordered_cols_r:
            if c not in ref_pivot.columns:
                ref_pivot[c] = 0.0
        missing_cols_r = [c for c in ref_pivot.columns if c not in ordered_cols_r and c != 'Referee Name']
        ordered_cols_r = ['Referee Name'] + comps_sorted + missing_cols_r
        ref_pivot = ref_pivot[ordered_cols_r]
        ref_pivot['Total'] = ref_pivot[[c for c in ref_pivot.columns if c not in ('Referee Name', 'Total')]].sum(axis=1)
        ref_pivot = ref_pivot.sort_values(['Total', 'Referee Name'], ascending=[False, True])

        # Export to Excel with reference formatting from finalupdatedcode.py (lines 176-283)
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Write data frames
            df.to_excel(writer, sheet_name='Upcoming Matches', index=False)
            ump_pivot.to_excel(writer, sheet_name='Umpire Days', index=False)
            ref_pivot.to_excel(writer, sheet_name='Referee Days', index=False)

            # Create formats exactly like reference
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#FFFF00',
                'valign': 'vcenter',
                'align': 'center',
                'text_wrap': True
            })
            data_format = workbook.add_format({
                'valign': 'vcenter',
                'align': 'center',
                'right': 1,
                'num_format': '0.0'
            })
            top_border = workbook.add_format({'top': 2})
            bottom_border = workbook.add_format({'bottom': 2})
            right_border = workbook.add_format({'right': 2})
            left_border = workbook.add_format({'left': 2})

            def format_worksheet(worksheet, df_local):
                # First column left-aligned width 15; others centered width 10 with data_format
                worksheet.set_column(0, 0, 15, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
                worksheet.set_column(1, len(df_local.columns) - 1, 10, data_format)
                # Headers with yellow background + heavy border box
                for col_num, value in enumerate(df_local.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.conditional_format(0, col_num, 0, col_num, {
                        'type': 'no_blanks',
                        'format': workbook.add_format({
                            'bold': True, 'bg_color': '#FFFF00', 'valign': 'vcenter', 'align': 'center',
                            'top': 2, 'bottom': 2, 'left': 2, 'right': 2, 'text_wrap': True
                        })
                    })
                # Borders for data area
                max_row = len(df_local)
                max_col = len(df_local.columns) - 1
                worksheet.conditional_format(1, 0, max_row, max_col, {'type': 'no_blanks', 'format': right_border})
                worksheet.conditional_format(1, 0, max_row, max_col, {'type': 'no_blanks', 'format': left_border})
                worksheet.conditional_format(max_row, 0, max_row, max_col, {'type': 'no_blanks', 'format': bottom_border})
                # Hide gridlines and unused areas
                worksheet.hide_gridlines(2)
                worksheet.set_default_row(hide_unused_rows=True)
                worksheet.set_selection(max_row + 1, 0, max_row + 1, max_col + 1)
                worksheet.set_row(max_row + 2, None, None, {'hidden': True})
                worksheet.set_column(max_col + 2, 16383, None, None, {'hidden': True})

            # Apply formatting to all three sheets
            format_worksheet(writer.sheets['Upcoming Matches'], df)
            format_worksheet(writer.sheets['Umpire Days'], ump_pivot)
            format_worksheet(writer.sheets['Referee Days'], ref_pivot)

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


# ===============================
# Streamlit real-time scraping UI
# ===============================
if 'st' in globals():
    @st.cache_data(show_spinner=False)
    def _cached_competitions() -> dict:
        s = BCCIUpcomingMatchesScraper(platforms=["domestic"], genders=["men","women"])
        comps = s.discover_competitions_from_competition_js()
        # Fallback to API discovery if JSONP fails
        if not comps:
            merged = {}
            for gender in ("men","women"):
                part = s.discover_competitions_from_api("domestic", gender)
                for cid, meta in part.items():
                    merged[cid] = meta
            comps = merged
        return comps or {}

    def _format_comp_label(cid: int, comps: dict) -> str:
        meta = comps.get(cid, {})
        return f"{meta.get('name','Competition')} (ID: {cid})"

    def _to_date(s):
        if pd.isna(s) or s == "":
            return None
        try:
            return pd.to_datetime(s, errors="coerce").date()
        except Exception:
            return None

    def _pivot_umpires(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d['Number of Days'] = pd.to_numeric(d.get('Number of Days', 0), errors='coerce').fillna(0.0)
        base = ['CompetitionName', 'Number of Days']
        u1 = d[base + ['GroundUmpire1']].rename(columns={'GroundUmpire1': 'Umpire Name', 'CompetitionName': 'Tournament Name'})
        u2 = d[base + ['GroundUmpire2']].rename(columns={'GroundUmpire2': 'Umpire Name', 'CompetitionName': 'Tournament Name'})
        umps = pd.concat([u1,u2], ignore_index=True)
        umps['Umpire Name'] = umps['Umpire Name'].fillna('').astype(str).str.strip()
        umps = umps[umps['Umpire Name'] != '']
        comps_sorted = sorted([c for c in umps['Tournament Name'].dropna().unique().tolist() if c])
        pv = umps.pivot_table(index='Umpire Name', columns='Tournament Name', values='Number of Days', aggfunc='sum', fill_value=0.0)
        pv = pv.reindex(columns=comps_sorted, fill_value=0.0)
        pv['Total'] = pv.sum(axis=1)
        pv = pv.sort_values('Total', ascending=False)
        return pv.reset_index()

    def _pivot_referees(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d['Number of Days'] = pd.to_numeric(d.get('Number of Days', 0), errors='coerce').fillna(0.0)
        ref = d[['CompetitionName', 'Number of Days', 'Referee']].rename(columns={'Referee': 'Referee Name', 'CompetitionName': 'Tournament Name'})
        ref['Referee Name'] = ref['Referee Name'].fillna('').astype(str).str.strip()
        ref = ref[ref['Referee Name'] != '']
        comps_sorted = sorted([c for c in ref['Tournament Name'].dropna().unique().tolist() if c])
        pv = ref.pivot_table(index='Referee Name', columns='Tournament Name', values='Number of Days', aggfunc='sum', fill_value=0.0)
        pv = pv.reindex(columns=comps_sorted, fill_value=0.0)
        pv['Total'] = pv.sum(axis=1)
        pv = pv.sort_values('Total', ascending=False)
        return pv.reset_index()

    def _write_formatted_excel(buf: BytesIO, df_main: pd.DataFrame, ump_pivot: pd.DataFrame, ref_pivot: pd.DataFrame):
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df_main.to_excel(writer, sheet_name='Upcoming Matches', index=False)
            ump_pivot.to_excel(writer, sheet_name='Umpire Days', index=False)
            ref_pivot.to_excel(writer, sheet_name='Referee Days', index=False)
            wb = writer.book
            header = wb.add_format({'bold': True, 'bg_color': '#FFFF00', 'valign': 'vcenter', 'align': 'center', 'text_wrap': True})
            datafmt = wb.add_format({'valign': 'vcenter', 'align': 'center', 'right': 1, 'num_format': '0.0'})
            borders_top = wb.add_format({'top': 2}); borders_bot = wb.add_format({'bottom': 2})
            borders_right = wb.add_format({'right': 2}); borders_left = wb.add_format({'left': 2})

            def _auto_fit(ws, df_local: pd.DataFrame):
                # Auto-fit widths by content (min 10; first column min 15)
                for j, col in enumerate(df_local.columns):
                    series = df_local[col].astype(str).fillna("")
                    max_len = max([len(str(col))] + [len(s) for s in series.tolist()])
                    width = max_len * 1.2
                    width = max(width, 15 if j == 0 else 10)
                    width = min(width, 60)
                    ws.set_column(j, j, width, datafmt if j>0 else wb.add_format({'align':'left','valign':'vcenter'}))

            def _fmt(ws, df_local: pd.DataFrame):
                # Header row with heavy border box
                for col_num, value in enumerate(df_local.columns.values):
                    ws.write(0, col_num, value, header)
                    ws.conditional_format(0, col_num, 0, col_num, {'type': 'no_blanks', 'format': wb.add_format({'bold': True,'bg_color': '#FFFF00','valign': 'vcenter','align': 'center','top': 2,'bottom': 2,'left': 2,'right': 2,'text_wrap': True})})
                max_row = len(df_local); max_col = max(0, len(df_local.columns)-1)
                ws.conditional_format(1,0,max_row,max_col,{'type':'no_blanks','format': borders_right})
                ws.conditional_format(1,0,max_row,max_col,{'type':'no_blanks','format': borders_left})
                ws.conditional_format(max_row,0,max_row,max_col,{'type':'no_blanks','format': borders_bot})
                ws.hide_gridlines(2)
                ws.set_default_row(hide_unused_rows=True)
                ws.set_selection(max_row+1,0,max_row+1,max_col+1)
                ws.set_row(max_row+2, None, None, {'hidden': True})
                ws.set_column(max_col+2, 16383, None, None, {'hidden': True})
                _auto_fit(ws, df_local)

            _fmt(writer.sheets['Upcoming Matches'], df_main)
            _fmt(writer.sheets['Umpire Days'], ump_pivot)
            _fmt(writer.sheets['Referee Days'], ref_pivot)

    def run_streamlit_app():
        st.set_page_config(page_title="BCCI Real-time Scraper", layout="wide")
        st.title("🏏 BCCI Cricket Real-time Scraping Dashboard")
        st.caption("Discover competitions, scrape live from BCCI feeds, filter and analyze, and download formatted Excel.")

        # Phase 1: Discover competitions
        with st.spinner("Discovering competitions..."):
            comps = _cached_competitions()
        if not comps:
            st.error("No competitions discovered. Please retry later.")
            return
        comp_ids_sorted = sorted(comps.keys())

        # Sidebar selection
        st.sidebar.header("Competition Selection")
        select_all = st.sidebar.checkbox("Select All Competitions", value=True)
        options = comp_ids_sorted
        default = options if select_all else []
        selected = st.sidebar.multiselect(
            "Competitions",
            options=options,
            default=default,
            format_func=lambda cid: _format_comp_label(cid, comps),
            help="Select competitions to scrape"
        )
        if select_all and len(selected) != len(options):
            # Keep select all in sync
            selected = options

        # Phase 2: Scraping interface
        st.sidebar.header("Scraping")
        start_btn = st.sidebar.button("🚀 Start Scraping", type="primary")
        status = st.empty(); prog = st.progress(0)
        if start_btn:
            scraper = BCCIUpcomingMatchesScraper(platforms=["domestic"], genders=["men","women"])
            scraper.competitions = {cid: comps[cid] for cid in selected} if selected else {cid: comps[cid] for cid in comp_ids_sorted}
            total = len(scraper.competitions)
            matches_total = 0
            for i, cid in enumerate(scraper.competitions.keys()):
                name = scraper.competitions[cid].get('name', f'Competition {cid}')
                status.info(f"({i+1}/{total}) Fetching: {name} (ID: {cid})")
                try:
                    matches_total += scraper.fetch_competition_data(cid)
                except Exception as e:
                    st.warning(f"Failed {cid}: {e}")
                prog.progress((i+1)/max(1,total))
                time.sleep(0.8)
            status.success(f"Done. Competitions processed: {total}. Matches found: {matches_total}.")

            # Convert to DataFrame and store in session
            df = pd.DataFrame(scraper.upcoming_matches)
            if not df.empty:
                desired = ['Platform','CompetitionGender','CompetitionID','CompetitionName','MatchStatus','MatchOrder','MatchID','MatchTypeID','MatchType','Number of Days','MatchName','MatchDate','GroundUmpire1','GroundUmpire2','ThirdUmpire','Referee','GroundName']
                df = df[[c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]]
                df['MatchDate'] = df['MatchDate'].apply(_to_date)
                df['Number of Days'] = pd.to_numeric(df.get('Number of Days', 0), errors='coerce').fillna(0.0)
            st.session_state['raw_df'] = df

        # Phase 3: Filters & Analysis
        df_raw = st.session_state.get('raw_df')
        if df_raw is None or df_raw.empty:
            st.info("Scrape some data to enable filtering and analysis.")
            return
        st.sidebar.header("Filters")
        mt_all = sorted(df_raw['MatchType'].dropna().unique().tolist())
        mo_all = sorted(df_raw['MatchOrder'].dropna().unique().tolist())
        g_all = sorted(df_raw['CompetitionGender'].dropna().unique().tolist())
        dates = df_raw['MatchDate'].dropna()
        dmin = dates.min() if not dates.empty else None
        dmax = dates.max() if not dates.empty else None
        if 'filters_init' not in st.session_state:
            st.session_state['f_mt'] = mt_all
            st.session_state['f_mo'] = mo_all
            st.session_state['f_g'] = g_all
            st.session_state['f_range'] = (dmin, dmax)
            st.session_state['filters_init'] = True
        if st.sidebar.button("Clear Filters"):
            st.session_state['f_mt'] = mt_all
            st.session_state['f_mo'] = mo_all
            st.session_state['f_g'] = g_all
            st.session_state['f_range'] = (dmin, dmax)
        sel_mt = st.sidebar.multiselect("Match Type", options=mt_all, default=st.session_state['f_mt'])
        sel_mo = st.sidebar.multiselect("Match Order", options=mo_all, default=st.session_state['f_mo'])
        sel_g  = st.sidebar.multiselect("Gender", options=g_all, default=st.session_state['f_g'])
        dr = st.sidebar.date_input("Date Range", value=st.session_state['f_range'] if all(st.session_state['f_range']) else None, min_value=dmin, max_value=dmax)
        if isinstance(dr, tuple) and len(dr)==2:
            st.session_state['f_range'] = dr
        st.session_state['f_mt'] = sel_mt; st.session_state['f_mo'] = sel_mo; st.session_state['f_g'] = sel_g

        with st.spinner("Applying filters..."):
            df = df_raw.copy()
            if sel_mt: df = df[df['MatchType'].isin(sel_mt)]
            if sel_mo: df = df[df['MatchOrder'].isin(sel_mo)]
            if sel_g:  df = df[df['CompetitionGender'].isin(sel_g)]
            if all(st.session_state['f_range']):
                start, end = st.session_state['f_range']
                df = df[(df['MatchDate'] >= start) & (df['MatchDate'] <= end)]

        total_matches = len(df_raw); filtered_matches = len(df)
        st.caption(f"Filter Summary: Showing {filtered_matches} of {total_matches} matches")
        c1, c2, c3 = st.columns(3)
        c1.metric("Matches", f"{filtered_matches}")
        c2.metric("Competitions", f"{df['CompetitionName'].nunique()}")
        uniq_officials = pd.unique(pd.concat([df['GroundUmpire1'], df['GroundUmpire2'], df['Referee']]).dropna()).size
        c3.metric("Officials (unique)", f"{uniq_officials}")

        st.subheader("📊 Match Details")
        st.write(f"Rows: {len(df)}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("👨‍⚖️ Umpire Workload Analysis")
        with st.spinner("Computing umpire pivot..."):
            ump_pivot = _pivot_umpires(df)
        st.dataframe(ump_pivot, use_container_width=True, hide_index=True)

        st.subheader("🏏 Referee Workload Analysis")
        with st.spinner("Computing referee pivot..."):
            ref_pivot = _pivot_referees(df)
        st.dataframe(ref_pivot, use_container_width=True, hide_index=True)

        st.subheader("📥 Download Filtered Data")
        if df.empty:
            st.caption("Nothing to download – adjust filters.")
        else:
            out = BytesIO()
            _write_formatted_excel(out, df, ump_pivot, ref_pivot)
            out.seek(0)
            st.download_button(
                label="Download Excel",
                data=out,
                file_name="bcci_filtered_scraped.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

if __name__ == "__main__":
    # If executed via Streamlit, run the Streamlit app; otherwise run CLI main
    if 'st' in globals() and os.environ.get('STREAMLIT_SERVER_PORT'):
        run_streamlit_app()
    else:
        main()
