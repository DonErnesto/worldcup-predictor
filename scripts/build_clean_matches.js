const fs = require("fs");
const path = require("path");

const RAW_ROOT = path.join("data", "raw", "openfootball-worldcup", "worldcup-master");
const RANKINGS_ROOT = path.join("data", "raw", "fifa-rankings", "by_schedule");
const OUT_PATH = path.join("data", "processed", "world_cup_matches.csv");

const TEAM_ALIASES = new Map([
  ["Bosnia & Herzegovina", "Bosnia and Herzegovina"],
  ["Cape Verde", "Cabo Verde"],
  ["Czech Republic", "Czechia"],
  ["DR Congo", "Congo DR"],
  ["Iran", "IR Iran"],
  ["Ivory Coast", "Côte d'Ivoire"],
  ["South Korea", "Korea Republic"],
  ["Turkey", "Türkiye"],
  ["United States", "USA"],
]);

const MONTHS = new Map([
  ["Jan", "01"], ["January", "01"],
  ["Feb", "02"], ["February", "02"],
  ["Mar", "03"], ["March", "03"],
  ["Apr", "04"], ["April", "04"],
  ["May", "05"],
  ["Jun", "06"], ["June", "06"],
  ["Jul", "07"], ["July", "07"],
  ["Aug", "08"], ["August", "08"],
  ["Sep", "09"], ["September", "09"],
  ["Oct", "10"], ["October", "10"],
  ["Nov", "11"], ["November", "11"],
  ["Dec", "12"], ["December", "12"],
]);

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function normalizeStage(stageRaw) {
  const raw = (stageRaw || "").toLowerCase();
  if (raw.includes("group") && raw.includes("play-off")) return "group_playoff";
  if (raw.startsWith("group")) return "group";
  if (raw.includes("first stage")) return "group";
  if (raw.includes("final round")) return "final_group";
  if (raw.includes("round of 32")) return "round_of_32";
  if (raw.includes("round of 16")) return "round_of_16";
  if (raw.includes("quarter")) return "quarter_final";
  if (raw.includes("semi")) return "semi_final";
  if (raw.includes("third") || raw.includes("third place")) return "third_place";
  if (raw === "final" || raw.includes(" final")) return "final";
  if (raw.includes("preliminary")) return "preliminary_round";
  if (raw.includes("first round")) return "first_round";
  return raw.replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
}

function isKnockout(stage) {
  return !["group", "group_playoff", "final_group"].includes(stage);
}

function parseDateLine(line, year) {
  const clean = line.trim();
  const match = clean.match(/^(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+)?([A-Za-z]+)\s+(\d{1,2})\b/);
  if (!match) return null;
  const month = MONTHS.get(match[1]);
  if (!month) return null;
  return `${year}-${month}-${String(match[2]).padStart(2, "0")}`;
}

function stripLeadIn(text) {
  return text
    .trim()
    .replace(/^\(\d+\)\s*/, "")
    .replace(/^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Za-z]+\s+\d{1,2}\s+(?:\d{1,2}:\d{2}\s+)?/, "")
    .replace(/^[A-Za-z]+\s+\d{1,2}\s+(?:\d{1,2}:\d{2}\s+)?/, "")
    .replace(/^\d{1,2}:\d{2}(?:\s+UTC[+-]\d+)?\s+/, "")
    .trim();
}

function consumeScoreDetails(rest) {
  let details = "";
  let remainder = rest.trim();
  let aet = false;
  let goalsA90 = null;
  let goalsB90 = null;
  let penaltiesA = null;
  let penaltiesB = null;

  const aetMatch = remainder.match(/^a\.?e\.?t\.?/i);
  if (aetMatch) {
    aet = true;
    details += aetMatch[0];
    remainder = remainder.slice(aetMatch[0].length).trim();
  }

  const parenMatch = remainder.match(/^\(([^)]*)\)/);
  if (parenMatch) {
    details += `${details ? " " : ""}${parenMatch[0]}`;
    const firstScore = parenMatch[1].match(/(\d+)\s*-\s*(\d+)/);
    if (aet && firstScore) {
      goalsA90 = Number(firstScore[1]);
      goalsB90 = Number(firstScore[2]);
    }
    remainder = remainder.slice(parenMatch[0].length).trim();
  }

  if (remainder.startsWith(",")) {
    details += `${details ? " " : ""},`;
    remainder = remainder.slice(1).trim();
  }

  const penMatch = remainder.match(/^(\d+)\s*-\s*(\d+)\s+pen\.?/i);
  if (penMatch) {
    penaltiesA = Number(penMatch[1]);
    penaltiesB = Number(penMatch[2]);
    details += `${details ? " " : ""}${penMatch[0]}`;
    remainder = remainder.slice(penMatch[0].length).trim();
  }

  return { details, remainder, aet, goalsA90, goalsB90, penaltiesA, penaltiesB };
}

function parsePlayedMatch(beforeVenue) {
  const before = stripLeadIn(beforeVenue);
  const fixtureThenScore = before.match(/^(.+?)\s+v\s+(.+?)\s+(\d+)\s*-\s*(\d+)(.*)$/);
  if (fixtureThenScore) {
    const details = consumeScoreDetails(fixtureThenScore[5]);
    const mainGoalsA = Number(fixtureThenScore[3]);
    const mainGoalsB = Number(fixtureThenScore[4]);
    return {
      country_a: fixtureThenScore[1].trim(),
      country_b: fixtureThenScore[2].trim(),
      goals_a_90: details.aet ? details.goalsA90 : mainGoalsA,
      goals_b_90: details.aet ? details.goalsB90 : mainGoalsB,
      goals_a_after_extra_time: details.aet ? mainGoalsA : null,
      goals_b_after_extra_time: details.aet ? mainGoalsB : null,
      penalties_a: details.penaltiesA,
      penalties_b: details.penaltiesB,
      extra_time: details.aet,
      score_details: details.details,
    };
  }

  const score = before.match(/\b(\d+)\s*-\s*(\d+)\b/);
  if (!score) return null;

  const teamA = before.slice(0, score.index).trim();
  const mainGoalsA = Number(score[1]);
  const mainGoalsB = Number(score[2]);
  const afterScore = before.slice(score.index + score[0].length);
  const details = consumeScoreDetails(afterScore);
  const teamB = details.remainder.trim();
  if (!teamA || !teamB) return null;

  const goalsA90 = details.aet ? details.goalsA90 : mainGoalsA;
  const goalsB90 = details.aet ? details.goalsB90 : mainGoalsB;

  return {
    country_a: teamA,
    country_b: teamB,
    goals_a_90: goalsA90,
    goals_b_90: goalsB90,
    goals_a_after_extra_time: details.aet ? mainGoalsA : null,
    goals_b_after_extra_time: details.aet ? mainGoalsB : null,
    penalties_a: details.penaltiesA,
    penalties_b: details.penaltiesB,
    extra_time: details.aet,
    score_details: details.details,
  };
}

function parseFixture(beforeVenue) {
  const before = stripLeadIn(beforeVenue);
  const fixture = before.match(/^(.+?)\s+v\s+(.+)$/);
  if (!fixture) return null;
  return {
    country_a: fixture[1].trim(),
    country_b: fixture[2].trim(),
    goals_a_90: null,
    goals_b_90: null,
    goals_a_after_extra_time: null,
    goals_b_after_extra_time: null,
    penalties_a: null,
    penalties_b: null,
    extra_time: false,
    score_details: "",
  };
}

function outcome(goalsA, goalsB) {
  if (goalsA === null || goalsB === null || goalsA === undefined || goalsB === undefined) return "";
  if (goalsA > goalsB) return "A_WIN";
  if (goalsA < goalsB) return "B_WIN";
  return "DRAW";
}

function teamLookupName(name) {
  return TEAM_ALIASES.get(name) || name;
}

function loadRankings() {
  const rankingsByYear = new Map();
  if (!fs.existsSync(RANKINGS_ROOT)) return rankingsByYear;

  for (const file of fs.readdirSync(RANKINGS_ROOT)) {
    const yearMatch = file.match(/^(\d{4})_/);
    if (!yearMatch) continue;
    const year = Number(yearMatch[1]);
    const json = JSON.parse(fs.readFileSync(path.join(RANKINGS_ROOT, file), "utf8"));
    const byName = new Map();
    for (const row of json.Results || []) {
      for (const name of row.TeamName || []) {
        byName.set(name.Description, row);
      }
    }
    rankingsByYear.set(year, byName);
  }
  return rankingsByYear;
}

function rankingFor(rankingsByYear, year, team) {
  const rankingYear = rankingsByYear.has(year) ? year : null;
  if (!rankingYear) return null;
  const table = rankingsByYear.get(rankingYear);
  return table.get(teamLookupName(team)) || null;
}

function parseTournamentFile(filePath, tournamentYear) {
  const rows = [];
  let stageRaw = "";
  let matchDate = "";

  for (const rawLine of fs.readFileSync(filePath, "utf8").split(/\r?\n/)) {
    const line = rawLine.replace(/\t/g, " ");
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("=")) continue;

    if (trimmed.startsWith("▪")) {
      const nextStage = trimmed.replace(/^▪+\s*/, "").split("|")[0].trim();
      if (!/^Matchday\b/i.test(nextStage)) stageRaw = nextStage;
      continue;
    }

    const parsedDate = parseDateLine(trimmed, tournamentYear);
    if (parsedDate) matchDate = parsedDate;

    if (!line.includes("@")) continue;
    const [beforeVenue, ...venueParts] = line.split("@");
    const venue = venueParts.join("@").trim();
    const parsed = /\b\d+\s*-\s*\d+\b/.test(beforeVenue)
      ? parsePlayedMatch(beforeVenue)
      : parseFixture(beforeVenue);
    if (!parsed) continue;

    const stage = normalizeStage(stageRaw);
    rows.push({
      tournament_year: tournamentYear,
      game_type: "championship",
      stage_raw: stageRaw,
      stage,
      is_knockout: isKnockout(stage),
      match_date: matchDate,
      venue,
      ...parsed,
      outcome_90: outcome(parsed.goals_a_90, parsed.goals_b_90),
    });
  }

  return rows;
}

function buildRows() {
  const rankingsByYear = loadRankings();
  const rows = [];

  const tournamentDirs = fs.readdirSync(RAW_ROOT)
    .filter((name) => /^\d{4}--/.test(name))
    .sort();

  for (const dir of tournamentDirs) {
    const year = Number(dir.slice(0, 4));
    for (const file of ["cup.txt", "cup_finals.txt"]) {
      const filePath = path.join(RAW_ROOT, dir, file);
      if (fs.existsSync(filePath)) rows.push(...parseTournamentFile(filePath, year));
    }
  }

  return rows.map((row, index) => {
    const rankA = rankingFor(rankingsByYear, row.tournament_year, row.country_a);
    const rankB = rankingFor(rankingsByYear, row.tournament_year, row.country_b);
    return {
      match_id: index + 1,
      ...row,
      country_a_code: rankA?.IdCountry || "",
      country_b_code: rankB?.IdCountry || "",
      rank_a: rankA?.Rank || "",
      rank_b: rankB?.Rank || "",
      ranking_points_a: rankA?.DecimalTotalPoints || "",
      ranking_points_b: rankB?.DecimalTotalPoints || "",
      rank_diff: rankA && rankB ? rankB.Rank - rankA.Rank : "",
      ranking_points_diff: rankA && rankB ? Number((rankA.DecimalTotalPoints - rankB.DecimalTotalPoints).toFixed(2)) : "",
      confederation_a: rankA?.ConfederationName || "",
      confederation_b: rankB?.ConfederationName || "",
      same_confederation: rankA && rankB ? rankA.ConfederationName === rankB.ConfederationName : "",
    };
  });
}

const rows = buildRows();
const columns = [
  "match_id",
  "tournament_year",
  "game_type",
  "stage_raw",
  "stage",
  "is_knockout",
  "match_date",
  "country_a",
  "country_b",
  "country_a_code",
  "country_b_code",
  "goals_a_90",
  "goals_b_90",
  "outcome_90",
  "goals_a_after_extra_time",
  "goals_b_after_extra_time",
  "penalties_a",
  "penalties_b",
  "extra_time",
  "score_details",
  "venue",
  "rank_a",
  "rank_b",
  "ranking_points_a",
  "ranking_points_b",
  "rank_diff",
  "ranking_points_diff",
  "confederation_a",
  "confederation_b",
  "same_confederation",
];

fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
fs.writeFileSync(
  OUT_PATH,
  `${columns.join(",")}\n${rows.map((row) => columns.map((column) => csvEscape(row[column])).join(",")).join("\n")}\n`,
);

const played = rows.filter((row) => row.outcome_90).length;
const fixtures = rows.length - played;
console.log(`wrote ${rows.length} rows to ${OUT_PATH}`);
console.log(`played=${played} fixtures=${fixtures}`);
