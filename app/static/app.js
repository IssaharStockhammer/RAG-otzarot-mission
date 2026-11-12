const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const savedQuerySelect = document.getElementById("saved-query");
const previousButton = document.getElementById("previous-query");
const nextButton = document.getElementById("next-query");
const queryCounter = document.getElementById("query-counter");
const searchButton = document.getElementById("search-button");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const workspaceEl = document.querySelector(".workspace");
const analysisPanelEl = document.getElementById("analysis-panel");
const analysisTitleEl = document.getElementById("analysis-title");
const analysisDescriptionEl = document.getElementById("analysis-description");
const analysisFeedbackListEl = document.getElementById("analysis-feedback-list");
const stickyQueryTextEl = document.getElementById("sticky-query-text");
const minConfidenceInput = document.getElementById("min-confidence");
const minConfidenceValue = document.getElementById("min-confidence-value");
const lexicalWeightInput = document.getElementById("lexical-weight");
const denseWeightInput = document.getElementById("dense-weight");
const lexicalWeightValue = document.getElementById("lexical-weight-value");
const denseWeightValue = document.getElementById("dense-weight-value");
const weightTuningEl = document.getElementById("weight-tuning");
const weightNoteEl = document.getElementById("weight-note");
const resetWeightsButton = document.getElementById("reset-weights");
const applyWeightsButton = document.getElementById("apply-weights");
const devSummaryGridEl = document.getElementById("dev-summary-grid");
const refreshFeedbackSummaryButton = document.getElementById("refresh-feedback-summary");

let savedQueries = [];
let currentMode = new URLSearchParams(window.location.search).get("dev") === "1" ? "text" : "user";
let lastSearchContext = null;
let lastSearchPayload = null;
let semanticAvailable = false;
const STATUS_CLASSES = ["status-info", "status-success", "status-warning", "status-error"];
const SCORE_HINTS = {
  "ציון כולל": "שקלול אותות ההתאמה לטקסט.",
  "יישור": "איכות התאמת הפסקה לעמוד לפי ה-OCR.",
  "מילולי": "חפיפת מילים וביטויים לשאלה.",
  "סמנטי": "דמיון במשמעות גם ללא אותן מילים.",
  "פער מוביל": "הפער בין המועמד הראשון לבא אחריו.",
};
const DEFAULT_WEIGHTS = { lexical: 0.45, semantic: 0.55 };

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatScore(value, digits = 3) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "-";
}

function setStatus(message, type = "info") {
  statusEl.textContent = message;
  statusEl.classList.remove(...STATUS_CLASSES);
  statusEl.classList.add(`status-${type}`);
}

function selectedQueryIndex() {
  return savedQueries.findIndex((query) => query.id === savedQuerySelect.value);
}

function updateQueryNavigation() {
  const index = selectedQueryIndex();
  const hasSelection = index >= 0;
  previousButton.disabled = !hasSelection || index === 0;
  nextButton.disabled = !hasSelection || index === savedQueries.length - 1;
  queryCounter.textContent = hasSelection
    ? `שאלה ${index + 1} מתוך ${savedQueries.length} · ${savedQueries[index].id}`
    : "שאלה חופשית";
}

function setStickyQuery(text) {
  const normalized = String(text || "").trim();
  stickyQueryTextEl.textContent = normalized || "—";
  stickyQueryTextEl.title = normalized || "—";
}

function normalizedWeights(lexical, semantic) {
  const total = lexical + semantic;
  if (total <= 0) {
    return { lexical: 0, semantic: 0 };
  }
  return { lexical: lexical / total, semantic: semantic / total };
}

function getSearchWeights() {
  if (currentMode !== "text") {
    return { ...DEFAULT_WEIGHTS };
  }
  if (!semanticAvailable) {
    return { lexical: 1, semantic: 0 };
  }
  return normalizedWeights(
    Number(lexicalWeightInput.value),
    Number(denseWeightInput.value),
  );
}

function updateWeightDisplay() {
  const weights = getSearchWeights();
  lexicalWeightValue.textContent = `${Math.round(weights.lexical * 100)}%`;
  denseWeightValue.textContent = `${Math.round(weights.semantic * 100)}%`;
  if (lastSearchContext) {
    setStatus("המשקולות השתנו; התוצאות מוצגות מהריצה הקודמת. הרץ שוב כדי לראות את השפעתן.", "warning");
  }
  localStorage.setItem("lexicalWeight", lexicalWeightInput.value);
  localStorage.setItem("denseWeight", denseWeightInput.value);
}

async function checkSemanticAvailability() {
  try {
    const response = await fetch("/api/index-status");
    if (!response.ok) {
      throw new Error(`בדיקת זמינות חיפוש סמנטי נכשלה (${response.status})`);
    }
    const payload = await response.json();
    semanticAvailable = payload.deepinfra_query_embeddings_enabled === true;
    denseWeightInput.disabled = !semanticAvailable;
    weightNoteEl.textContent = semanticAvailable
      ? "המשקולות מנורמלות ל־100%. הרץ שוב את אותה שאלה כדי להשוות."
      : "חיפוש סמנטי אינו פעיל כרגע; רק ההתאמה המילולית תשפיע.";
    updateWeightDisplay();
  } catch (error) {
    setStatus(error.message || "שגיאה בבדיקת זמינות חיפוש סמנטי.", "error");
  }
}

function scoreHue(score) {
  return Math.round(Math.max(0, Math.min(1, Number(score) || 0)) * 120);
}

function scoreMetric(label, value) {
  const numeric = Number(value);
  const hint = SCORE_HINTS[label] || "";
  const available = Number.isFinite(numeric);
  const score = available ? Math.max(0, Math.min(1, numeric)) : 0;
  return `
    <div class="score-metric">
      <div class="score-header">
        <span class="score-header-main">${escapeHtml(label)} <span class="score-help" title="${escapeHtml(hint)}">ⓘ</span></span>
        <strong>${available ? formatScore(score) : "—"}</strong>
      </div>
      <p class="score-caption">${escapeHtml(hint)}</p>
      <div class="score-track">
        <div class="score-fill" style="width:${available ? score * 100 : 0}%;background:hsl(${scoreHue(score)} 78% 42%)"></div>
      </div>
    </div>
  `;
}

function feedbackOptions() {
  return '<option value="" selected disabled>בחר דירוג</option>' + [5, 4, 3, 2, 1, 0]
    .map((score) => `<option value="${score}">${score}${score === 5 ? " — מצוין" : score === 0 ? " — לא רלוונטי" : ""}</option>`)
    .join("");
}

function renderFeedbackControls(result, index, mode) {
  const ratingField = mode === "text" ? "text_relevance" : "location_accuracy";
  const ratingLabel = mode === "text" ? "רלוונטיות הטקסט (0–5)" : "דיוק המיקום (0–5)";
  const locationFields = mode === "location"
    ? `
      <label>העמוד נכון?
        <select data-feedback-field="page_correct">
          <option value="" selected disabled>בחר</option>
          <option value="yes">כן</option><option value="partial">חלקית</option><option value="no">לא</option>
        </select>
      </label>
      <label>טווח הסימון נכון?
        <select data-feedback-field="location_band_correct">
          <option value="" selected disabled>בחר</option>
          <option value="yes">כן</option><option value="partial">חלקית</option><option value="no">לא</option>
        </select>
      </label>`
    : "";
  return `
    <article class="candidate-feedback-item" data-feedback-index="${index}">
      <h4>מועמד ${result.rank} · ${escapeHtml(result.page)}</h4>
      <label>${ratingLabel}
        <select data-feedback-field="${ratingField}">${feedbackOptions()}</select>
      </label>
      ${locationFields}
      <label>הערה
        <textarea rows="2" data-feedback-field="comment" placeholder="הערה אופציונלית"></textarea>
      </label>
      <button type="button" class="save-feedback-btn" data-result-index="${index}">שמור משוב</button>
    </article>
  `;
}

function renderSummary(summary, mode) {
  if (mode === "text") {
    const count = summary.rating_counts?.text_relevance ?? 0;
    const average = summary.averages?.text_relevance;
    const hit = summary.kpis?.text_hit_rate_ge4;
    devSummaryGridEl.innerHTML = `
      <div class="dev-summary-item"><span class="source-key">דירוגי טקסט</span><span class="source-value">${count}</span></div>
      <div class="dev-summary-item"><span class="source-key">ממוצע טקסט</span><span class="source-value">${average == null ? "—" : formatScore(average)}</span></div>
      <div class="dev-summary-item"><span class="source-key">דירוג ≥4</span><span class="source-value">${hit == null ? "—" : `${Math.round(hit * 100)}%`}</span></div>`;
  } else {
    const count = summary.rating_counts?.location_accuracy ?? 0;
    const average = summary.averages?.location_accuracy;
    const hit = summary.kpis?.location_hit_rate_ge4;
    devSummaryGridEl.innerHTML = `
      <div class="dev-summary-item"><span class="source-key">דירוגי מיקום</span><span class="source-value">${count}</span></div>
      <div class="dev-summary-item"><span class="source-key">ממוצע מיקום</span><span class="source-value">${average == null ? "—" : formatScore(average)}</span></div>
      <div class="dev-summary-item"><span class="source-key">דירוג ≥4</span><span class="source-value">${hit == null ? "—" : `${Math.round(hit * 100)}%`}</span></div>`;
  }
}

async function loadFeedbackSummary() {
  if (currentMode === "user") {
    return;
  }
  try {
    const response = await fetch("/api/dev-feedback/summary");
    if (!response.ok) {
      throw new Error(`טעינת סיכום המשוב נכשלה (${response.status})`);
    }
    renderSummary(await response.json(), currentMode);
  } catch (error) {
    setStatus(error.message || "שגיאה בטעינת סיכום המשוב.", "error");
  }
}

function columnMarkup(result) {
  const highlight = result.highlight || {};
  const column = highlight.column_hint === "left" ? "left" : "right";
  const page = escapeHtml(result.page);
  const globalLeft = Number(highlight.left_pct);
  const globalWidth = Number(highlight.width_pct);
  const cropStart = column === "right" ? 50 : 0;
  const localLeft = Number.isFinite(globalLeft)
    ? Math.max(0, Math.min(100, (globalLeft - cropStart) * 2))
    : 5;
  const localWidth = Number.isFinite(globalWidth)
    ? Math.max(2, Math.min(100, globalWidth * 2))
    : 88;
  const marker = Number.isFinite(Number(highlight.top_pct)) &&
    Number.isFinite(Number(highlight.height_pct))
    ? `<div class="page-marker" style="top:${Number(highlight.top_pct)}%;height:${Number(highlight.height_pct)}%;--left:${localLeft}%;--width:${localWidth}%"></div>`
    : "";
  return `
    <div class="page-column-viewport" data-column="${column}">
      <img class="column-page-image" data-column="${column}" src="/pages/${page}.png" alt="עמוד ${page}, טור ${column === "right" ? "ימני" : "שמאלי"}" />
      ${marker}
      <span class="marker-label">טווח משוער · טור ${column === "right" ? "ימני" : "שמאלי"}</span>
    </div>
  `;
}

function configureColumnViews(root = resultsEl) {
  root.querySelectorAll(".page-column-viewport").forEach((viewport) => {
    const image = viewport.querySelector(".column-page-image");
    if (!(image instanceof HTMLImageElement)) {
      return;
    }
    const sizeViewport = () => {
      if (!image.naturalWidth || !image.naturalHeight) {
        return;
      }
      const halfPageRatio = image.naturalWidth / 2 / image.naturalHeight;
      const availableWidth = Math.max(160, viewport.clientWidth);
      const maxHeight = Math.min(window.innerHeight * 0.78, 900);
      const scale = Math.min(maxHeight / image.naturalHeight, availableWidth / (image.naturalWidth / 2));
      viewport.style.width = `${image.naturalWidth / 2 * scale}px`;
      viewport.style.height = `${image.naturalHeight * scale}px`;
      viewport.style.aspectRatio = String(halfPageRatio);
    };
    if (image.complete) {
      sizeViewport();
    } else {
      image.addEventListener("load", sizeViewport, { once: true });
    }
  });
}

function renderResult(result, index) {
  const details = result.score_details || {};
  const confidence = Math.max(0, Math.min(1, Number(result.confidence) || 0));
  const sourceCard = `
    <section class="sources-card">
      <h3>מקורות</h3>
      <div class="sources-grid">
        <div class="source-item"><span class="source-key">טקסט</span><span class="source-value">${escapeHtml(result.text_source || result.source)}</span></div>
        <div class="source-item"><span class="source-key">OCR למיפוי</span><span class="source-value">${escapeHtml(result.source)}</span></div>
        <div class="source-item"><span class="source-key">מראה מקום</span><span class="source-value">${escapeHtml(result.external_ref || "—")}</span></div>
        <div class="source-item"><span class="source-key">יישור למקור</span><span class="source-value">${escapeHtml(result.alignment_status || "—")} · ${formatScore(result.alignment_score, 2)}</span></div>
      </div>
    </section>`;
  const scoreCard = `
    <section class="score-card">
      <h3>ציוני מערכת</h3>
      <p class="score-caption">משקלי הריצה: מילולי ${formatScore(details.lexical_weight, 2)} · סמנטי ${formatScore(details.dense_weight, 2)}.</p>
      <div class="score-grid">
        ${scoreMetric("ציון כולל", confidence)}
        ${scoreMetric("יישור", details.alignment)}
        ${scoreMetric("מילולי", details.lexical)}
        ${scoreMetric("סמנטי", details.dense)}
      </div>
    </section>`;
  const locationText = `
    <div class="result-meta">
      <span class="badge">עמוד ${escapeHtml(result.page)}</span>
      <span class="badge">${escapeHtml(result.line_from ?? "—")}–${escapeHtml(result.line_to ?? "—")} שורות OCR משוערות</span>
    </div>
    <p class="location-text">${escapeHtml(result.location_hint)}</p>`;

  let content = "";
  if (currentMode === "text") {
    content = `
      <section class="full-text-card">
        <h3>הטקסט המלא של המקטע שאוחזר</h3>
        <p class="full-text" dir="rtl">${escapeHtml(result.full_text || result.snippet_or_location)}</p>
      </section>
      ${sourceCard}${scoreCard}`;
  } else if (currentMode === "location") {
    content = `${locationText}${columnMarkup(result)}${sourceCard}`;
  } else {
    content = `
      ${locationText}
      <section class="snippet-card"><h3>קטע מהממצא</h3><p class="snippet" dir="rtl">${escapeHtml(result.snippet_or_location)}</p></section>
      ${columnMarkup(result)}
      ${sourceCard}`;
  }

  return `
    <article class="result-card" id="candidate-${index}">
      <header class="result-header">
        <span class="rank-pill">מועמד ${result.rank}</span>
        <span class="confidence-pill" style="--score-hue:${scoreHue(confidence)}">ציון ${formatScore(confidence)}</span>
      </header>
      <div class="result-main">${content}</div>
    </article>`;
}

function updateMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".mode-tab").forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });

  const isAnalysis = mode !== "user";
  analysisPanelEl.classList.toggle("hidden", !isAnalysis);
  workspaceEl.classList.toggle("analysis-hidden", !isAnalysis);
  weightTuningEl.classList.toggle("hidden", mode !== "text");
  analysisTitleEl.textContent = mode === "text"
    ? "בדיקת התאמת טקסט"
    : mode === "location" ? "בדיקת זיהוי המיקום" : "ניתוח התוצאות";
  analysisDescriptionEl.textContent = mode === "text"
    ? "דרג רק את רלוונטיות הטקסט; המשקולות משפיעות על סדר המועמדים."
    : mode === "location" ? "דרג רק את נכונות העמוד והטור המסומן." : "";
  if (mode !== "user") {
    loadFeedbackSummary();
    if (mode === "text") {
      checkSemanticAvailability();
    }
  }
  renderCurrentResults();
}

function renderCurrentResults() {
  if (!lastSearchPayload || lastSearchPayload.not_found || !lastSearchPayload.results?.length) {
    analysisFeedbackListEl.innerHTML = '<p class="empty-note">הרץ חיפוש כדי לדרג את המועמדים.</p>';
    return;
  }
  resultsEl.innerHTML = lastSearchPayload.results
    .map((result, index) => renderResult(result, index))
    .join("");

  if (currentMode === "user") {
    analysisFeedbackListEl.replaceChildren();
  } else {
    analysisFeedbackListEl.innerHTML = lastSearchPayload.results
      .map((result, index) => renderFeedbackControls(result, index, currentMode))
      .join("");
  }
  configureColumnViews();
}

function selectQuery(index) {
  const query = savedQueries[index];
  if (!query) {
    return;
  }
  savedQuerySelect.value = query.id;
  queryInput.value = query.query;
  setStickyQuery(query.query);
  updateQueryNavigation();
  setStatus("מוכן לחיפוש.", "info");
  lastSearchContext = null;
  lastSearchPayload = null;
  resultsEl.innerHTML = `
    <div class="empty-state"><span class="empty-icon" aria-hidden="true">⌕</span>
    <h2>השאלה נבחרה</h2><p>הרץ חיפוש כדי לראות מועמדים.</p></div>`;
  analysisFeedbackListEl.innerHTML = '<p class="empty-note">הרץ חיפוש כדי לדרג את המועמדים.</p>';
}

async function loadSavedQueries() {
  savedQuerySelect.disabled = true;
  try {
    const response = await fetch("/api/queries");
    if (!response.ok) {
      throw new Error(`טעינת השאלות נכשלה (${response.status})`);
    }
    savedQueries = await response.json();
    if (!Array.isArray(savedQueries) || savedQueries.length === 0) {
      throw new Error("קובץ השאלות לא מכיל שאלות זמינות.");
    }
    savedQuerySelect.replaceChildren(
      ...savedQueries.map((query) => {
        const option = document.createElement("option");
        option.value = query.id;
        option.textContent = `${query.id} — ${query.query}`;
        return option;
      }),
    );
    savedQuerySelect.disabled = false;
    selectQuery(0);
  } catch (error) {
    savedQuerySelect.replaceChildren(new Option("לא ניתן לטעון שאלות", ""));
    setStatus(error.message || "שגיאה בטעינת קובץ השאלות.", "error");
  }
}

async function runSearch() {
  const query = queryInput.value.trim();
  setStickyQuery(query);
  const minConfidence = Number(minConfidenceInput.value);
  const rankingWeights = getSearchWeights();
  if (!query) {
    setStatus("יש לבחור שאלה או להקליד שאלה לפני החיפוש.", "warning");
    queryInput.focus();
    return;
  }
  if (rankingWeights.lexical + rankingWeights.semantic <= 0) {
    setStatus("יש להגדיר משקל חיובי לפחות לאחד מאותות החיפוש.", "warning");
    return;
  }
  searchButton.disabled = true;
  setStatus("מחפש במאגר...", "info");
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: 3,
        min_confidence: minConfidence,
        lexical_weight: rankingWeights.lexical,
        dense_weight: rankingWeights.semantic,
      }),
    });
    if (!response.ok) {
      throw new Error(`החיפוש נכשל (${response.status})`);
    }
    const payload = await response.json();
    lastSearchPayload = payload;
    lastSearchContext = {
      queryId: savedQuerySelect.value || null,
      queryText: query,
      retrievalMode: payload.retrieval_mode,
      minConfidence,
      rankingWeights: payload.ranking_weights,
    };
    if (payload.not_found || !payload.results?.length) {
      resultsEl.innerHTML = '<div class="empty-state"><h2>לא נמצאה התאמה</h2><p>בדוק את סף הקירבה או נסה ניסוח אחר.</p></div>';
      analysisFeedbackListEl.innerHTML = '<p class="empty-note">אין מועמדים לדירוג.</p>';
      setStatus(
        payload.not_found_reason === "below_min_confidence"
          ? `לא מצאנו התאמה בציון הנדרש (${formatScore(minConfidence, 2)}).`
          : "לא נמצאה התאמה רלוונטית.",
        "warning",
      );
      return;
    }
    const modeLabel = payload.retrieval_mode === "hybrid (DeepInfra embeddings + lexical)"
      ? "סמנטי + מילולי"
      : "מילולי בלבד";
    setStatus(
      `נמצאו ${payload.results.length} מועמדים · ${modeLabel} · משקל מילולי ${formatScore(payload.ranking_weights.lexical, 2)} / סמנטי ${formatScore(payload.ranking_weights.semantic, 2)}.`,
      "success",
    );
    renderCurrentResults();
    if (currentMode !== "user") {
      loadFeedbackSummary();
    }
  } catch (error) {
    setStatus(error.message || "אירעה שגיאה בזמן החיפוש.", "error");
  } finally {
    searchButton.disabled = false;
  }
}

async function saveFeedback(button) {
  if (!lastSearchContext || !lastSearchPayload?.results) {
    return;
  }
  const index = Number(button.dataset.resultIndex);
  const result = lastSearchPayload.results[index];
  const card = button.closest(".candidate-feedback-item");
  if (!result || !card) {
    return;
  }
  const value = (field) => card.querySelector(`[data-feedback-field="${field}"]`)?.value;
  const isTextMode = currentMode === "text";
  const ratingField = isTextMode ? "text_relevance" : "location_accuracy";
  if (!value(ratingField)) {
    setStatus("בחר דירוג לפני שמירת המשוב.", "warning");
    return;
  }
  const pageStatus = value("page_correct");
  const payload = {
    query_id: lastSearchContext.queryId,
    query_text: lastSearchContext.queryText,
    retrieval_mode: lastSearchContext.retrievalMode,
    min_confidence: lastSearchContext.minConfidence,
    rank: Number(result.rank),
    page: result.page,
    text_source: result.text_source || result.source,
    ocr_source: result.source,
    text_relevance: isTextMode ? Number(value("text_relevance")) : null,
    location_accuracy: isTextMode ? null : Number(value("location_accuracy")),
    page_correct: isTextMode ? null : pageStatus === "yes" ? true : pageStatus === "no" ? false : null,
    location_band_correct: isTextMode ? null : value("location_band_correct"),
    comment: String(value("comment") || "").trim() || null,
    ranking_weights: lastSearchContext.rankingWeights,
    score_snapshot: {
      confidence: result.confidence,
      ...(result.score_details || {}),
    },
  };
  try {
    button.disabled = true;
    const response = await fetch("/api/dev-feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errorPayload = await response.json();
      throw new Error(errorPayload.detail || `שמירת משוב נכשלה (${response.status})`);
    }
    button.textContent = "נשמר ✓";
    setStatus("המשוב נשמר יחד עם המשקולות ששימשו לחיפוש.", "success");
    await loadFeedbackSummary();
  } catch (error) {
    setStatus(error.message || "שגיאה בשמירת המשוב.", "error");
    button.disabled = false;
  }
}

savedQuerySelect.addEventListener("change", () => {
  const index = selectedQueryIndex();
  if (index >= 0) {
    selectQuery(index);
  }
});

previousButton.addEventListener("click", () => {
  const index = selectedQueryIndex();
  if (index > 0) {
    selectQuery(index - 1);
  }
});

nextButton.addEventListener("click", () => {
  const index = selectedQueryIndex();
  if (index >= 0 && index < savedQueries.length - 1) {
    selectQuery(index + 1);
  }
});

queryInput.addEventListener("input", () => setStickyQuery(queryInput.value));
minConfidenceInput.addEventListener("input", () => {
  minConfidenceValue.textContent = formatScore(minConfidenceInput.value, 2);
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
});
document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => updateMode(button.dataset.mode));
});
analysisFeedbackListEl.addEventListener("click", (event) => {
  const button = event.target.closest(".save-feedback-btn");
  if (button) {
    saveFeedback(button);
  }
});
refreshFeedbackSummaryButton.addEventListener("click", loadFeedbackSummary);
lexicalWeightInput.addEventListener("input", updateWeightDisplay);
denseWeightInput.addEventListener("input", updateWeightDisplay);
resetWeightsButton.addEventListener("click", () => {
  lexicalWeightInput.value = String(DEFAULT_WEIGHTS.lexical);
  denseWeightInput.value = String(DEFAULT_WEIGHTS.semantic);
  updateWeightDisplay();
});
applyWeightsButton.addEventListener("click", runSearch);
window.addEventListener("resize", () => configureColumnViews());

lexicalWeightInput.value = localStorage.getItem("lexicalWeight") || String(DEFAULT_WEIGHTS.lexical);
denseWeightInput.value = localStorage.getItem("denseWeight") || String(DEFAULT_WEIGHTS.semantic);
minConfidenceValue.textContent = formatScore(minConfidenceInput.value, 2);
updateWeightDisplay();
setStickyQuery(queryInput.value);
loadSavedQueries();
updateMode(currentMode);
