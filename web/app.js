const form = document.querySelector("#incident-form");
const statusEl = document.querySelector("#status");
const selectedResults = document.querySelector("#selected-results");
const allScores = document.querySelector("#all-scores");
const predictionCount = document.querySelector("#prediction-count");
const sampleButton = document.querySelector("#sample-button");
const clearButton = document.querySelector("#clear-button");
const activeModel = document.querySelector("#active-model");
const datasetStats = document.querySelector("#dataset-stats");
const integrityStats = document.querySelector("#integrity-stats");
const modelScoreRows = document.querySelector("#model-score-rows");
const topIssues = document.querySelector("#top-issues");
const topJurisdictions = document.querySelector("#top-jurisdictions");
const topSectors = document.querySelector("#top-sectors");
const notebookTakeaways = document.querySelector("#notebook-takeaways");

const samples = [
  {
    headline: "Facial recognition system misidentifies a passenger at an airport",
    purpose: "Identity verification",
    technology: "Facial recognition",
    sector: "Public sector",
    jurisdiction: "United States",
    deployer: "Airport authority",
    developer: "",
    system_name: "",
    news_trigger: "Misidentification",
  },
  {
    headline: "Chatbot gives unsafe medical advice to patients seeking urgent care",
    purpose: "Medical support",
    technology: "Generative AI",
    sector: "Healthcare",
    jurisdiction: "European Union",
    deployer: "Hospital network",
    developer: "",
    system_name: "",
    news_trigger: "Harmful output",
  },
  {
    headline: "School monitoring software flags students unfairly during remote exams",
    purpose: "Exam proctoring",
    technology: "Behavior analytics",
    sector: "Education",
    jurisdiction: "United Kingdom",
    deployer: "University",
    developer: "",
    system_name: "",
    news_trigger: "Disputed decision",
  },
];

let sampleIndex = 0;

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function fieldValue(name) {
  return form.elements[name].value.trim();
}

function setStatus(label, state) {
  statusEl.textContent = label;
  statusEl.dataset.state = state;
}

function setLoading(isLoading) {
  form.querySelectorAll("button").forEach((button) => {
    button.disabled = isLoading;
  });
}

function payloadFromForm() {
  return {
    model: form.elements.model.value,
    headline: fieldValue("headline"),
    purpose: fieldValue("purpose") || null,
    technology: fieldValue("technology") || null,
    deployer: fieldValue("deployer") || null,
    developer: fieldValue("developer") || null,
    system_name: fieldValue("system_name") || null,
    news_trigger: fieldValue("news_trigger") || null,
    jurisdiction: fieldValue("jurisdiction") || null,
    sector: fieldValue("sector") || null,
  };
}

function renderSelected(predictions) {
  predictionCount.textContent = predictions.length;
  selectedResults.innerHTML = "";

  if (!predictions.length) {
    selectedResults.innerHTML = '<div class="empty-state">No label crossed its tuned threshold.</div>';
    return;
  }

  predictions.forEach((item) => {
    const element = document.createElement("div");
    element.className = "tag-result";
    element.innerHTML = `
      <div class="tag-topline">
        <div class="tag-label"></div>
        <div class="tag-probability">${formatPercent(item.probability)}</div>
      </div>
      <div class="tag-threshold">Threshold ${formatPercent(item.threshold)}</div>
    `;
    element.querySelector(".tag-label").textContent = item.label;
    selectedResults.appendChild(element);
  });
}

function renderScores(scores) {
  allScores.innerHTML = "";

  scores.forEach((item) => {
    const row = document.createElement("div");
    row.className = "score-row";
    row.dataset.selected = item.selected;
    row.innerHTML = `
      <div class="score-name"></div>
      <div class="meter" aria-hidden="true"><div class="meter-fill"></div></div>
      <div class="score-value">${formatPercent(item.probability)} / ${formatPercent(item.threshold)}</div>
    `;
    row.querySelector(".score-name").textContent = item.label;
    row.querySelector(".meter-fill").style.width = formatPercent(item.probability);
    allScores.appendChild(row);
  });
}

function updateActiveModel(data) {
  if (data && data.model && data.model.name) {
    activeModel.textContent = data.model.name;
    return;
  }
  activeModel.textContent = form.elements.model.value === "tinybert" ? "TinyBERT" : "TF-IDF Logistic Regression";
}

async function predict() {
  const payload = payloadFromForm();
  if (!payload.headline) {
    form.elements.headline.focus();
    return;
  }

  setLoading(true);
  setStatus("Running", "loading");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Prediction failed with status ${response.status}`);
    }

    const data = await response.json();
    updateActiveModel(data);
    renderSelected(data.predictions);
    renderScores(data.scores);
    setStatus("Ready", "ready");
  } catch (error) {
    selectedResults.innerHTML = '<div class="empty-state">Prediction failed. Check the server logs.</div>';
    setStatus("Error", "error");
  } finally {
    setLoading(false);
  }
}

function applySample() {
  const sample = samples[sampleIndex % samples.length];
  sampleIndex += 1;
  Object.entries(sample).forEach(([key, value]) => {
    form.elements[key].value = value;
  });
}

function clearForm() {
  form.reset();
  form.querySelectorAll("input:not([type='radio']), textarea").forEach((field) => {
    field.value = "";
  });
  form.elements.model.value = "tfidf";
  predictionCount.textContent = "0";
  updateActiveModel();
  selectedResults.innerHTML = '<div class="empty-state">Run a prediction to show thresholded tags.</div>';
  allScores.innerHTML = "";
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    setStatus(data.ok ? "Ready" : "Missing", data.ok ? "ready" : "error");
  } catch (error) {
    setStatus("Error", "error");
  }
}

function renderStats(container, rows) {
  container.innerHTML = "";
  rows.forEach((item) => {
    const element = document.createElement("div");
    element.className = "stat-item";
    element.innerHTML = `
      <div class="stat-value"></div>
      <div class="stat-label"></div>
    `;
    element.querySelector(".stat-value").textContent = item.value;
    element.querySelector(".stat-label").textContent = item.label;
    container.appendChild(element);
  });
}

function renderBars(container, rows) {
  container.innerHTML = "";
  const max = Math.max(...rows.map((item) => item.count));
  rows.forEach((item) => {
    const element = document.createElement("div");
    element.className = "bar-row";
    element.innerHTML = `
      <div class="bar-row-top">
        <span class="bar-label"></span>
        <span class="bar-count">${item.count.toLocaleString()}</span>
      </div>
      <div class="bar-track" aria-hidden="true"><div class="bar-fill"></div></div>
    `;
    element.querySelector(".bar-label").textContent = item.label;
    element.querySelector(".bar-fill").style.width = `${Math.round((item.count / max) * 100)}%`;
    container.appendChild(element);
  });
}

function renderModelScores(rows) {
  modelScoreRows.innerHTML = "";
  rows.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td></td>
      <td></td>
      <td>${item.micro_f1.toFixed(4)}</td>
      <td>${item.macro_f1.toFixed(4)}</td>
    `;
    row.children[0].textContent = item.split;
    row.children[1].textContent = item.model;
    modelScoreRows.appendChild(row);
  });
}

function renderTakeaways(rows) {
  notebookTakeaways.innerHTML = "";
  rows.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    notebookTakeaways.appendChild(li);
  });
}

async function loadResults() {
  try {
    const response = await fetch("/api/results");
    if (!response.ok) {
      throw new Error(`Results request failed with status ${response.status}`);
    }
    const data = await response.json();
    renderStats(datasetStats, data.dataset);
    renderStats(integrityStats, data.integrity);
    renderModelScores(data.model_scores);
    renderBars(topIssues, data.top_issues);
    renderBars(topJurisdictions, data.top_jurisdictions);
    renderBars(topSectors, data.top_sectors);
    renderTakeaways(data.takeaways);
  } catch (error) {
    notebookTakeaways.innerHTML = "<li>Notebook results are unavailable.</li>";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  predict();
});

sampleButton.addEventListener("click", applySample);
clearButton.addEventListener("click", clearForm);
Array.from(form.elements.model).forEach((input) => {
  input.addEventListener("change", updateActiveModel);
});

checkHealth();
loadResults();
