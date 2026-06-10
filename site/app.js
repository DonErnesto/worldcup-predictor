const state = {
  payload: null,
  teamsByCode: new Map(),
};

const elements = {
  countryA: document.querySelector("#country-a"),
  countryB: document.querySelector("#country-b"),
  kicktipp: document.querySelector("#kicktipp-mode"),
  rankingDate: document.querySelector("#ranking-date"),
  warning: document.querySelector("#same-team-warning"),
  teamAName: document.querySelector("#team-a-name"),
  teamBName: document.querySelector("#team-b-name"),
  teamAMeta: document.querySelector("#team-a-meta"),
  teamBMeta: document.querySelector("#team-b-meta"),
  rawScore: document.querySelector("#raw-score"),
  selectedScore: document.querySelector("#selected-score"),
  historyPair: document.querySelector("#history-pair"),
  historyList: document.querySelector("#history-list"),
};

async function init() {
  const response = await fetch("data/predictions.json");
  if (!response.ok) {
    throw new Error(`Could not load prediction data: ${response.status}`);
  }
  state.payload = await response.json();
  state.teamsByCode = new Map(state.payload.teams.map((team) => [team.code, team]));
  populateTeams();
  bindEvents();
  render();
}

function populateTeams() {
  const options = state.payload.teams.map((team) => {
    const option = document.createElement("option");
    option.value = team.code;
    option.textContent = `${team.name} (${team.code})`;
    return option;
  });

  elements.countryA.replaceChildren(...options.map((option) => option.cloneNode(true)));
  elements.countryB.replaceChildren(...options.map((option) => option.cloneNode(true)));
  elements.countryA.value = state.payload.teams[0].code;
  elements.countryB.value = state.payload.teams[1].code;
  elements.rankingDate.textContent = `FIFA ranking ${formatDate(state.payload.metadata.ranking_snapshot_date)}`;
}

function bindEvents() {
  elements.countryA.addEventListener("change", render);
  elements.countryB.addEventListener("change", render);
  document.querySelectorAll("input[name='phase']").forEach((input) => {
    input.addEventListener("change", render);
  });
  elements.kicktipp.addEventListener("change", render);
}

function render() {
  const codeA = elements.countryA.value;
  const codeB = elements.countryB.value;
  const phase = document.querySelector("input[name='phase']:checked").value;
  const scoreSelector = elements.kicktipp.checked ? "kicktipp" : "standard";
  const teamA = state.teamsByCode.get(codeA);
  const teamB = state.teamsByCode.get(codeB);
  const sameTeam = codeA === codeB;

  elements.warning.hidden = !sameTeam;
  elements.teamAName.textContent = teamA.name;
  elements.teamBName.textContent = teamB.name;
  elements.teamAMeta.textContent = teamMeta(teamA);
  elements.teamBMeta.textContent = teamMeta(teamB);
  elements.historyPair.textContent = `${teamA.code} vs ${teamB.code}`;

  if (sameTeam) {
    elements.rawScore.textContent = "-";
    elements.selectedScore.textContent = "-";
    elements.historyList.replaceChildren(emptyHistory("Choose two different countries."));
    return;
  }

  const prediction = state.payload.predictions[`${codeA}|${codeB}|${phase}|${scoreSelector}`];
  elements.rawScore.textContent = `${prediction.expected_goals_a.toFixed(2)} - ${prediction.expected_goals_b.toFixed(2)}`;
  elements.selectedScore.textContent = `${prediction.selected_goals_a} - ${prediction.selected_goals_b}`;
  renderHistory(codeA, codeB);
}

function renderHistory(codeA, codeB) {
  const key = [codeA, codeB].sort().join("|");
  const matches = state.payload.head_to_head[key] || [];
  if (!matches.length) {
    elements.historyList.replaceChildren(emptyHistory("No previous World Cup meetings in the data."));
    return;
  }

  elements.historyList.replaceChildren(
    ...matches.slice(0, 3).map((match) => {
      const normalized = normalizeHistory(match, codeA, codeB);
      const item = document.createElement("li");

      const date = document.createElement("span");
      date.className = "history-date";
      date.textContent = `${match.tournament_year} · ${formatDate(match.match_date)}`;

      const teams = document.createElement("span");
      teams.innerHTML = `<strong>${normalized.countryA}</strong> vs <strong>${normalized.countryB}</strong><br><span class="history-stage">${titleCase(match.stage)}</span>`;

      const score = document.createElement("span");
      score.className = "history-score";
      score.textContent = normalized.score;

      item.append(date, teams, score);
      return item;
    })
  );
}

function normalizeHistory(match, codeA, codeB) {
  const teamA = state.teamsByCode.get(codeA);
  const teamB = state.teamsByCode.get(codeB);
  if (match.country_a_code === codeA && match.country_b_code === codeB) {
    return {
      countryA: teamA.name,
      countryB: teamB.name,
      score: `${match.goals_a_90}-${match.goals_b_90}`,
    };
  }
  return {
    countryA: teamA.name,
    countryB: teamB.name,
    score: `${match.goals_b_90}-${match.goals_a_90}`,
  };
}

function emptyHistory(message) {
  const item = document.createElement("li");
  item.className = "empty-state";
  item.textContent = message;
  return item;
}

function teamMeta(team) {
  return `Rank ${team.rank} · ${team.ranking_points.toFixed(2)} pts · ${team.confederation}`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(new Date(`${value}T00:00:00Z`));
}

function titleCase(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

init().catch((error) => {
  elements.historyList.replaceChildren(emptyHistory(error.message));
});
