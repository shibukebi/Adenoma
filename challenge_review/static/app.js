const state = {
  user: null,
  config: null,
  slides: [],
  selectedId: null,
  page: 1,
  pages: 1,
  filters: {},
  viewer: null,
  slideMetadata: null,
  searchTimer: null,
  wsiRequestId: 0,
  tileSource: null,
  magnifierViewer: null,
  magnifierEnabled: false,
  magnifierReady: false,
  magnifierSlideId: null,
  magnifierTarget: 20,
  magnifierPointer: null,
  magnifierFrame: null,
  magnifierLoadingTimer: null,
  magnifierLoadingVersion: 0,
  magnifierRequiredLevel: 0,
  challengeTaxonomy: null,
  challengeReview: null,
  challengeSlideEligible: false,
  challengeSaveTimer: null,
  challengeSaving: false,
  challengeSavePromise: null,
  roiDrawEnabled: false,
  roiOverlays: new Map(),
  selectedRoiId: null,
  roiPointerAction: null,
  roiSaveQueue: Promise.resolve(),
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (response.status === 401) {
    showLogin();
    throw new Error("登录已过期");
  }
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.style.background = isError ? "#9d3934" : "#22313c";
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.hidden = true; }, 2600);
}

function showLogin() {
  $("#app-view").hidden = true;
  $("#login-view").hidden = false;
  state.user = null;
}

async function enterApp(user) {
  state.user = user;
  $("#login-view").hidden = true;
  $("#app-view").hidden = false;
  $("#account-name").textContent = user.display_name;
  $("#admin-button").hidden = user.role !== "admin";
  lucide.createIcons();
  if (!state.challengeTaxonomy) {
    state.challengeTaxonomy = await api("/api/challenge-taxonomy");
    populateChallengeTaxonomy();
  }
  await Promise.all([loadProgress(), loadSlides()]);
}

const challengeLabels = {
  confirm_original: "Confirm original label", correct_original: "Correct original label",
  remains_ambiguous: "Remains ambiguous", focal_evidence: "Focal evidence",
  subtle_morphology: "Subtle morphology", heterogeneous_lesion: "Heterogeneous lesion",
  poor_orientation: "Poor orientation", limited_sampling: "Limited sampling",
  requires_high_magnification: "Requires high magnification",
  requires_multiple_regions: "Requires multiple regions", technical_artifact: "Technical artifact",
  competing_morphology: "Competing / misleading morphology",
  global_architecture_required: "Global architecture required", insufficient_tissue: "Insufficient tissue",
  relevant_evidence_absent: "Relevant evidence absent", genuine_diagnostic_ambiguity: "Genuine diagnostic ambiguity",
  technical_limitation: "Technical limitation", other: "Other",
  discriminative: "Discriminative", confirmatory: "Confirmatory", contradictory: "Contradictory",
  mimic_confounder: "Mimic / confounder", hgd_defining: "HGD-defining",
  assessability_quality: "Assessability / quality", weak: "Weak", moderate: "Moderate",
  strong: "Strong", decisive: "Decisive", supports_hp: "Supports HP over SSL",
  supports_ssl: "Supports SSL", supports_tsa: "Supports TSA", supports_ta: "Supports TA",
  supports_tva: "Supports TVA", supports_adenoma: "Supports adenoma",
  supports_inflammatory: "Supports inflammatory", supports_both: "Supports both / non-discriminative",
  conflicts_final: "Conflicts with final diagnosis", uncertain: "Uncertain",
  definite_hgd: "Definite HGD evidence", suspicious_insufficient: "Suspicious but insufficient",
  no_definite_hgd: "No definite HGD identified", not_assessable: "Not assessable",
  supports_final: "Supports final diagnosis", supports_alternative: "Supports alternative diagnosis",
  non_discriminative: "Non-discriminative", basal_crypt_dilation: "Basal crypt dilation",
  horizontal_crypt_growth: "Horizontal crypt growth", boot_l_shaped_crypt: "Boot / L-shaped crypt",
  asymmetric_crypt_proliferation: "Asymmetric crypt proliferation", surface_serration: "Surface serration",
  ectopic_crypt_formation: "Ectopic crypt formation", eosinophilic_cytoplasm: "Eosinophilic cytoplasm",
  pencillate_nuclei: "Pencillate nuclei", slit_like_serration: "Slit-like serration",
  serrated_architecture: "Serrated architecture", conventional_dysplasia: "Conventional dysplasia",
  crypt_elongation: "Crypt elongation", villous_architecture: "Villous architecture",
  tubular_architecture: "Tubular architecture", tubulovillous_architecture: "Tubulovillous architecture",
  adenomatous_dysplasia: "Adenomatous dysplasia", regenerative_atypia: "Regenerative atypia",
  lamina_propria_inflammation: "Lamina propria inflammation", erosion_ulceration: "Erosion / ulceration",
  complex_glandular_architecture: "Complex glandular architecture", cribriforming: "Cribriforming",
  marked_cytologic_atypia: "Marked cytologic atypia", loss_of_polarity: "Loss of polarity",
  luminal_necrosis: "Luminal necrosis", fragmentation: "Fragmentation", limited_tissue: "Limited tissue",
  cautery_artifact: "Cautery artifact", crypt_architecture: "Crypt architecture",
  cytologic_atypia: "Cytologic atypia", inflammation: "Inflammation",
  tissue_quality_limitation: "Tissue quality limitation",
  retain_challenge: "Retain as true challenge",
  pending_label_adjudication: "Possible label error - adjudication",
  exclude_label_error: "Recommend exclusion - label error",
  retained_challenge: "Confirmed challenge",
  eligible_for_exclusion: "Eligible for exclusion",
  unresolved: "Unresolved",
  unreviewed: "Unreviewed",
};
const challengeLabel = (value) => challengeLabels[value] || value;

function fillSelect(selector, values, includeBlank = false) {
  const select = $(selector);
  select.replaceChildren();
  if (includeBlank) select.add(new Option("请选择", ""));
  values.forEach((value) => select.add(new Option(challengeLabel(value), value)));
}

function populateChallengeTaxonomy() {
  const taxonomy = state.challengeTaxonomy;
  fillSelect("#challenge-lesion", taxonomy.lesion_diagnoses);
  fillSelect("#challenge-hgd", taxonomy.hgd_statuses);
  fillSelect("#challenge-label-action", taxonomy.label_actions);
  fillSelect("#challenge-disposition", taxonomy.challenge_dispositions);
  fillSelect("#challenge-primary", taxonomy.primary_challenges, true);
  fillSelect("#challenge-no-roi-reason", taxonomy.no_roi_reasons, true);
  $("#challenge-modifiers").replaceChildren(...taxonomy.difficulty_modifiers.map((code) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" name="challenge-modifier" value="${code}"><span>${challengeLabel(code)}</span>`;
    return label;
  }));
  $("#challenge-confidence").replaceChildren(...[1, 2, 3, 4, 5].map((value) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="challenge-confidence" value="${value}"${value === 3 ? " checked" : ""}><span>${value}</span>`;
    return label;
  }));
}

function populateLabels() {
  ["#filter-label", "#filter-consensus"].forEach((selector) => {
    const select = $(selector);
    state.config.classes.forEach((label) => select.add(new Option(label, label)));
  });
  const review = $("#review-label");
  state.config.classes.forEach((label) => review.add(new Option(label, label)));
}

async function initialize() {
  state.config = await api("/api/config");
  populateLabels();
  bindEvents();
  lucide.createIcons();
  try {
    const user = await api("/api/auth/me");
    await enterApp(user);
  } catch (_) {
    showLogin();
  }
}

function bindEvents() {
  $("#login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("#login-error").textContent = "";
    try {
      const user = await api("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username: $("#login-username").value, password: $("#login-password").value }),
      });
      $("#login-password").value = "";
      await enterApp(user);
    } catch (error) { $("#login-error").textContent = error.message; }
  });

  $("#logout-button").addEventListener("click", async () => { await api("/api/auth/logout", { method: "POST" }); showLogin(); });
  $("#account-button").addEventListener("click", () => togglePopover("#account-menu"));
  $("#export-menu-button").addEventListener("click", () => togglePopover("#export-menu"));
  document.addEventListener("click", (event) => {
    if (!event.target.closest("#account-button, #account-menu")) $("#account-menu").hidden = true;
    if (!event.target.closest("#export-menu-button, #export-menu")) $("#export-menu").hidden = true;
  });

  ["#filter-core", "#filter-resolution", "#filter-label", "#filter-consensus", "#filter-source", "#filter-fold", "#filter-status", "#filter-consistency", "#filter-sort"].forEach((selector) => {
    $(selector).addEventListener("change", () => { state.page = 1; loadSlides(); });
  });
  $("#filter-search").addEventListener("input", () => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(() => { state.page = 1; loadSlides(); }, 280);
  });
  $("#clear-filters").addEventListener("click", clearFilters);
  $("#page-previous").addEventListener("click", () => changePage(-1));
  $("#page-next").addEventListener("click", () => changePage(1));
  $("#previous-slide").addEventListener("click", () => moveSelection(-1));
  $("#next-slide").addEventListener("click", () => moveSelection(1));
  $("#zoom-in").addEventListener("click", () => state.viewer?.viewport.zoomBy(1.5));
  $("#zoom-out").addEventListener("click", () => state.viewer?.viewport.zoomBy(0.67));
  $("#zoom-home").addEventListener("click", () => state.viewer?.viewport.goHome());
  $("#magnifier-toggle").addEventListener("click", toggleMagnifier);
  $("#roi-draw-toggle").addEventListener("click", toggleRoiDraw);
  $$("#magnifier-controls button").forEach((button) => {
    button.addEventListener("click", () => setMagnifierTarget(Number(button.dataset.magnification)));
  });
  $("#viewer-fullscreen").addEventListener("click", () => state.viewer?.setFullScreen(!state.viewer.isFullPage()));
  $("#viewer").addEventListener("pointermove", handleMagnifierPointerMove);
  $("#viewer").addEventListener("pointerleave", hideMagnifierPanel);
  $("#review-form").addEventListener("submit", saveReview);
  $("#challenge-review-form").addEventListener("input", handleChallengeInput);
  $("#challenge-review-form").addEventListener("change", handleChallengeInput);
  $("#challenge-review-form").addEventListener("submit", submitChallengeReview);
  $("#challenge-primary").addEventListener("change", updateChallengeConditionalFields);
  $("#challenge-no-roi").addEventListener("change", updateChallengeConditionalFields);
  $("#challenge-no-roi-reason").addEventListener("change", updateChallengeConditionalFields);
  $("#challenge-lock-button").addEventListener("click", toggleChallengeLock);
  $("#viewer").addEventListener("pointerdown", startRoiDrawing, true);
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (event.key === "Escape" && state.roiDrawEnabled) {
      disableRoiDraw();
      return;
    }
    if (event.key === "Escape" && state.magnifierEnabled) {
      disableMagnifier();
      return;
    }
    if (event.key === "ArrowLeft") moveSelection(-1);
    if (event.key === "ArrowRight") moveSelection(1);
  });

  $("#admin-button").addEventListener("click", openAdmin);
  $("#admin-close").addEventListener("click", () => $("#admin-dialog").close());
  $("#create-user-form").addEventListener("submit", createUser);
}

function togglePopover(selector) {
  const element = $(selector);
  element.hidden = !element.hidden;
  lucide.createIcons();
}

function currentFilters() {
  const consistency = $("#filter-consistency").value;
  return {
    search: $("#filter-search").value.trim(),
    core_only: $("#filter-core").checked ? "true" : "",
    original_label: $("#filter-label").value,
    consensus_label: $("#filter-consensus").value,
    source: $("#filter-source").value,
    cv_fold: $("#filter-fold").value,
    review_status: $("#filter-status").value,
    resolution_status: $("#filter-resolution").value,
    consistency_min: consistency,
    sort: $("#filter-sort").value,
  };
}

function queryString(parameters) {
  const query = new URLSearchParams();
  Object.entries(parameters).forEach(([key, value]) => { if (value !== "" && value != null) query.set(key, value); });
  return query.toString();
}

async function loadSlides(preferredId = null) {
  const filters = currentFilters();
  state.filters = filters;
  try {
    const data = await api(`/api/slides?${queryString({ ...filters, page: state.page, page_size: 50 })}`);
    state.slides = data.items;
    state.pages = data.pages;
    $("#queue-total").textContent = data.total.toLocaleString();
    $("#page-text").textContent = `${data.page} / ${data.pages}`;
    $("#page-previous").disabled = data.page <= 1;
    $("#page-next").disabled = data.page >= data.pages;
    renderSlideList();
    const target = preferredId && state.slides.some((item) => item.slide_id === preferredId)
      ? preferredId
      : (state.selectedId && state.slides.some((item) => item.slide_id === state.selectedId) ? state.selectedId : state.slides[0]?.slide_id);
    if (target && target !== state.selectedId) await selectSlide(target);
    if (!target) clearSelection();
  } catch (error) { showToast(error.message, true); }
}

function renderSlideList() {
  const container = $("#slide-list");
  container.replaceChildren();
  state.slides.forEach((slide) => {
    const button = document.createElement("button");
    button.className = `slide-row${slide.slide_id === state.selectedId ? " active" : ""}`;
    button.dataset.slideId = slide.slide_id;
    button.innerHTML = `
      <span class="slide-name" title="${escapeHtml(slide.slide_id)}">${escapeHtml(slide.slide_id)}</span>
      <span class="agreement">${slide.consensus_wrong_count}/14</span>
      <span class="slide-meta">
        <span class="label-chip">${slide.original_label}</span>
        <span>→ ${slide.consensus_wrong_class}</span>
        <span>${slide.source} · F${slide.cv_fold}</span>
        <span class="status-badge status-${slide.challenge_review_status}">${challengeLabel(slide.resolution_status)}</span>
        ${slide.wsi_available ? "" : '<span title="WSI 不可用">WSI ×</span>'}
      </span>`;
    button.addEventListener("click", () => selectSlide(slide.slide_id));
    container.append(button);
  });
}

async function selectSlide(slideId) {
  await flushChallengeDraft();
  clearRoiOverlays();
  state.selectedId = slideId;
  renderSlideList();
  $("#no-selection").hidden = true;
  $("#review-content").hidden = false;
  try {
    const detail = await api(`/api/slides/${encodeURIComponent(slideId)}`);
    renderDetail(detail);
    const challengePromise = loadChallengeReview(detail.slide);
    await openWsi(detail.slide);
    await challengePromise;
    renderRoiOverlays();
    prefetchNextHpSlide();
  } catch (error) { showToast(error.message, true); }
}

function renderDetail(detail) {
  const slide = detail.slide;
  $("#slide-id").textContent = slide.slide_id;
  $("#original-label").textContent = slide.original_label;
  $("#consensus-label").textContent = slide.consensus_wrong_class;
  $("#consensus-count").textContent = `${slide.consensus_wrong_count}/14 · ${slide.wrong_configurations} configs wrong`;
  $("#source-fold").textContent = `${slide.source} / Fold ${slide.cv_fold}`;
  $("#pathology-type").textContent = `${slide.pathology_type || "-"}${slide.grade ? ` · ${slide.grade}` : ""}`;
  $("#wsi-status").textContent = slide.wsi_available ? slide.wsi_format.toUpperCase() : "不可用";
  const badge = $("#review-status-badge");
  const reviewStatus = detail.review?.status || "unreviewed";
  badge.className = `status-badge status-${reviewStatus}`;
  badge.textContent = statusText(reviewStatus);

  const body = $("#prediction-body");
  body.replaceChildren();
  detail.predictions.forEach((prediction) => {
    const row = document.createElement("tr");
    const firstClass = prediction.is_correct ? "prediction-correct" : "prediction-wrong";
    const secondLabel = prediction.second_predicted_label || "-";
    const secondConfidence = prediction.second_confidence == null
      ? "-"
      : `${(prediction.second_confidence * 100).toFixed(1)}%`;
    row.innerHTML = `
      <td>${prediction.model}</td>
      <td>${prediction.feature}</td>
      <td><span class="candidate-label ${firstClass}">${prediction.predicted_label}</span><span class="candidate-confidence">${(prediction.confidence * 100).toFixed(1)}%</span></td>
      <td><span class="candidate-label candidate-second">${secondLabel}</span><span class="candidate-confidence">${secondConfidence}</span></td>`;
    body.append(row);
  });
  fillReview(detail.review, slide.original_label);
  renderPeerSummary(detail.peer_summary);
}

async function loadChallengeReview(slide) {
  state.challengeSlideEligible = slide.wrong_configurations === 14;
  state.challengeReview = null;
  $("#challenge-review-form").hidden = !state.challengeSlideEligible;
  $("#challenge-ineligible").hidden = state.challengeSlideEligible;
  $("#roi-draw-toggle").disabled = !state.challengeSlideEligible;
  if (!state.challengeSlideEligible) {
    disableRoiDraw();
    return;
  }
  const data = await api(`/api/slides/${encodeURIComponent(slide.slide_id)}/challenge-review`);
  state.challengeReview = data.review || defaultChallengeReview(slide.original_label);
  fillChallengeReview(state.challengeReview);
}

function splitOriginalLabel(label) {
  const hgd = { SSLD: "SSL", TSAD: "TSA", TAD: "TA", TVAD: "TVA" };
  return { lesion: hgd[label] || (state.challengeTaxonomy.lesion_diagnoses.includes(label) ? label : "Other"), hgd: hgd[label] ? "Present" : "Absent" };
}

function defaultChallengeReview(originalLabel) {
  const original = splitOriginalLabel(originalLabel);
  return {
    id: null, version: 0, lesion_diagnosis: original.lesion, hgd_status: original.hgd,
    label_action: "confirm_original", challenge_disposition: "retain_challenge", primary_challenge: "", primary_challenge_other: "",
    modifiers: [], difficulty_note: "", expert_confidence: 3, no_localizable_evidence: 0,
    no_roi_reason: "", no_roi_reason_other: "", status: "draft", locked: false, rois: [], peer_summary: null,
  };
}

function fillChallengeReview(review) {
  $("#challenge-lesion").value = review.lesion_diagnosis;
  $("#challenge-hgd").value = review.hgd_status;
  $("#challenge-label-action").value = review.label_action;
  $("#challenge-disposition").value = review.challenge_disposition || "retain_challenge";
  $("#challenge-primary").value = review.primary_challenge;
  $("#challenge-primary-other").value = review.primary_challenge_other || "";
  $("#challenge-note").value = review.difficulty_note || "";
  $$(`input[name='challenge-confidence']`).forEach((input) => { input.checked = Number(input.value) === review.expert_confidence; });
  const modifiers = new Set(review.modifiers || []);
  $$(`input[name='challenge-modifier']`).forEach((input) => { input.checked = modifiers.has(input.value); });
  $("#challenge-no-roi").checked = Boolean(review.no_localizable_evidence);
  $("#challenge-no-roi-reason").value = review.no_roi_reason || "";
  $("#challenge-no-roi-other").value = review.no_roi_reason_other || "";
  $("#challenge-save-state").textContent = review.status === "submitted" ? `已提交 · v${review.version}` : `草稿 · v${review.version}`;
  updateChallengeConditionalFields();
  renderRoiEditors();
  renderChallengePeerSummary(review.peer_summary);
  setChallengeLocked(Boolean(review.locked));
}

function challengePayload() {
  return {
    version: state.challengeReview?.version || 0,
    lesion_diagnosis: $("#challenge-lesion").value,
    hgd_status: $("#challenge-hgd").value,
    label_action: $("#challenge-label-action").value,
    challenge_disposition: $("#challenge-disposition").value,
    primary_challenge: $("#challenge-primary").value,
    primary_challenge_other: $("#challenge-primary-other").value,
    modifiers: $$(`input[name='challenge-modifier']:checked`).map((input) => input.value),
    difficulty_note: $("#challenge-note").value,
    expert_confidence: Number($(`input[name='challenge-confidence']:checked`)?.value || 3),
    no_localizable_evidence: $("#challenge-no-roi").checked,
    no_roi_reason: $("#challenge-no-roi-reason").value,
    no_roi_reason_other: $("#challenge-no-roi-other").value,
  };
}

function handleChallengeInput(event) {
  if (event?.target?.closest(".roi-editor")) return;
  if (!state.challengeSlideEligible || state.challengeReview?.locked) return;
  updateChallengeConditionalFields();
  scheduleChallengeDraft();
}

function scheduleChallengeDraft() {
  clearTimeout(state.challengeSaveTimer);
  $("#challenge-save-state").textContent = "有未保存修改";
  state.challengeSaveTimer = setTimeout(saveChallengeDraft, 800);
}

async function ensureChallengeReview() {
  if (state.challengeReview?.id) return state.challengeReview;
  if (state.challengeSaveTimer) await flushChallengeDraft();
  if (state.challengeSaving && state.challengeSavePromise) await state.challengeSavePromise;
  if (state.challengeReview?.id) return state.challengeReview;
  return saveChallengeDraft();
}

async function saveChallengeDraft() {
  if (!state.selectedId || !state.challengeSlideEligible || state.challengeReview?.locked) return state.challengeReview;
  if (state.challengeSaving && state.challengeSavePromise) return state.challengeSavePromise;
  clearTimeout(state.challengeSaveTimer);
  state.challengeSaveTimer = null;
  state.challengeSaving = true;
  const slideId = state.selectedId;
  const operation = (async () => {
    await state.roiSaveQueue.catch(() => {});
    const review = await api(`/api/slides/${encodeURIComponent(slideId)}/challenge-review/draft`, {
      method: "PUT", body: JSON.stringify(challengePayload()),
    });
    if (state.selectedId === slideId) {
      state.challengeReview = review;
      $("#challenge-save-state").textContent = `${review.status === "submitted" ? "已提交" : "草稿"} · v${review.version}`;
      renderRoiEditors();
    }
    return review;
  })();
  state.challengeSavePromise = operation;
  try {
    return await operation;
  } catch (error) {
    showToast(error.message, true);
    if (error.message.includes("another session")) await reloadChallengeReview();
    throw error;
  } finally {
    if (state.challengeSavePromise === operation) state.challengeSavePromise = null;
    state.challengeSaving = false;
  }
}

async function flushChallengeDraft() {
  if (state.challengeSaveTimer) {
    clearTimeout(state.challengeSaveTimer);
    state.challengeSaveTimer = null;
    try { await saveChallengeDraft(); } catch (_) {}
  } else if (state.challengeSaving && state.challengeSavePromise) {
    try { await state.challengeSavePromise; } catch (_) {}
  }
}

async function reloadChallengeReview() {
  if (!state.selectedId || !state.challengeSlideEligible) return;
  const data = await api(`/api/slides/${encodeURIComponent(state.selectedId)}/challenge-review`);
  state.challengeReview = data.review || state.challengeReview;
  fillChallengeReview(state.challengeReview);
  renderRoiOverlays();
}

async function submitChallengeReview(event) {
  event.preventDefault();
  try {
    await flushChallengeDraft();
    const review = await api(`/api/slides/${encodeURIComponent(state.selectedId)}/challenge-review/submit`, {
      method: "POST", body: JSON.stringify(challengePayload()),
    });
    state.challengeReview = review;
    fillChallengeReview(review);
    await loadProgress();
    showToast("Challenge Review 已提交");
  } catch (error) { showToast(error.message, true); }
}

function updateChallengeConditionalFields() {
  $("#challenge-primary-other-label").hidden = $("#challenge-primary").value !== "Other differential";
  $("#no-roi-fields").hidden = !$("#challenge-no-roi").checked;
  $("#challenge-no-roi-other-label").hidden = $("#challenge-no-roi-reason").value !== "other";
  if ($("#challenge-no-roi").checked) disableRoiDraw();
  updateDispositionHint();
  renderRoiEditors();
}

function challengeFinalClass() {
  const lesion = $("#challenge-lesion").value;
  const hgd = $("#challenge-hgd").value;
  const highGrade = { SSL: "SSLD", TSA: "TSAD", TA: "TAD", TVA: "TVAD" };
  if (hgd === "Present" && highGrade[lesion]) return highGrade[lesion];
  if (hgd === "Absent" && ["HP", "SSL", "TSA", "TA", "TVA", "IP", "USA"].includes(lesion)) return lesion;
  return null;
}

function updateDispositionHint() {
  const hint = $("#challenge-disposition-hint");
  const finalLabel = challengeFinalClass();
  const original = $("#original-label").textContent.trim();
  const consensus = $("#consensus-label").textContent.trim();
  const disposition = $("#challenge-disposition");
  const qualifiesAsLabelError = Boolean(finalLabel && finalLabel === consensus && finalLabel !== original);
  [...disposition.options].forEach((option) => {
    if (["pending_label_adjudication", "exclude_label_error"].includes(option.value)) {
      option.disabled = !qualifiesAsLabelError;
    }
  });
  if (!qualifiesAsLabelError && disposition.value !== "retain_challenge") {
    disposition.value = "retain_challenge";
  }
  if (qualifiesAsLabelError && disposition.value !== "retain_challenge") {
    $("#challenge-label-action").value = "correct_original";
  }
  hint.className = "disposition-hint";
  if (!finalLabel) {
    hint.textContent = "当前诊断不能唯一映射为 11 类标签，建议保留为存疑或继续仲裁。";
    hint.classList.add("ambiguous");
  } else if (finalLabel === consensus && finalLabel !== original) {
    hint.textContent = `人工最终标签 ${finalLabel} 与模型共识一致、与原标签 ${original} 不同。可标记为标签问题；至少两位审核者一致后才进入可排除列表。`;
    hint.classList.add("model-match");
  } else if (finalLabel === original) {
    hint.textContent = `人工最终标签与原标签 ${original} 一致，当前病例仍属于模型共同失败的真实 challenge。`;
    hint.classList.add("original-match");
  } else {
    hint.textContent = `人工最终标签 ${finalLabel} 同时不同于原标签 ${original} 和模型共识 ${consensus}，建议标记为待仲裁，不自动排除。`;
    hint.classList.add("ambiguous");
  }
}

function renderChallengePeerSummary(summary) {
  const element = $("#challenge-peer-summary");
  if (!summary) { element.hidden = true; return; }
  const renderVotes = (items) => items.map((item) => `${challengeLabel(item.value)}:${item.votes}`).join(" · ");
  const resolution = challengeLabel(summary.resolution_status);
  const agreement = summary.agreed_revised_label ? ` · ${summary.agreed_revised_label} (${summary.agreed_reviewer_count} reviewers)` : "";
  element.innerHTML = `<strong>已提交审核汇总 (${summary.review_count})</strong><br>Resolution ${resolution}${agreement}<br>Diagnosis ${renderVotes(summary.lesion_votes)}<br>HGD ${renderVotes(summary.hgd_votes)}<br>Disposition ${renderVotes(summary.disposition_votes)}<br>Challenge ${renderVotes(summary.challenge_votes)}<br>ROI count ${renderVotes(summary.roi_count_distribution)}`;
  element.hidden = false;
}

function setChallengeLocked(locked) {
  const form = $("#challenge-review-form");
  [...form.elements].forEach((element) => { if (element.id !== "challenge-lock-button") element.disabled = locked; });
  $("#roi-draw-toggle").disabled = locked || !state.challengeSlideEligible;
  const banner = $("#challenge-lock-banner");
  banner.hidden = !locked;
  banner.textContent = locked ? "该审核已由管理员锁定，当前为只读状态。" : "";
  const button = $("#challenge-lock-button");
  button.hidden = state.user?.role !== "admin" || !state.challengeReview?.id || state.challengeReview?.status !== "submitted";
  button.textContent = locked ? "解锁" : "锁定";
  disableRoiDraw();
}

async function toggleChallengeLock() {
  const review = state.challengeReview;
  if (!review?.id) return;
  const action = review.locked ? "unlock" : "lock";
  try {
    state.challengeReview = await api(`/api/admin/challenge-reviews/${review.id}/${action}`, {
      method: "POST", body: JSON.stringify({ version: review.version }),
    });
    fillChallengeReview(state.challengeReview);
  } catch (error) { showToast(error.message, true); }
}

function fillReview(review, originalLabel) {
  $("#review-label").value = review?.revised_label || originalLabel;
  const statusValue = review?.status || "completed";
  const confidence = review?.diagnostic_confidence || "medium";
  $$("input[name='review-status']").forEach((input) => { input.checked = input.value === statusValue; });
  $$("input[name='review-confidence']").forEach((input) => { input.checked = input.value === confidence; });
  const flags = new Set(review?.quality_flags || []);
  $$("input[name='quality-flag']").forEach((input) => { input.checked = flags.has(input.value); });
  $("#review-notes").value = review?.notes || "";
  $("#saved-at").textContent = review ? `已保存 ${formatTime(review.updated_at)}` : "";
}

function renderPeerSummary(summary) {
  const element = $("#peer-summary");
  if (!summary) { element.hidden = true; return; }
  const votes = summary.label_votes.map((item) => `${item.revised_label}:${item.votes}`).join(" · ");
  element.textContent = `审核者汇总 (${summary.review_count})  ${votes}`;
  element.hidden = false;
}

async function openWsi(slide) {
  const requestId = ++state.wsiRequestId;
  state.viewer?.close();
  resetMagnifierSlide();
  $("#viewer-empty").hidden = true;
  $("#viewer-error").hidden = true;
  $("#viewer-loading").hidden = false;
  $("#viewer-loading span:last-child").textContent = slide.source === "hp"
    ? "正在初始化 iSyntax，首次打开可能需要 1–2 分钟"
    : "正在读取 WSI";
  $("#zoom-indicator").hidden = true;
  if (!slide.wsi_available) {
    showViewerError("原始 WSI 不可用");
    return;
  }
  try {
    if (slide.source === "hp") warmSlide(slide.slide_id);
    const metadata = await api(`/api/slides/${encodeURIComponent(slide.slide_id)}/metadata`);
    if (requestId !== state.wsiRequestId || slide.slide_id !== state.selectedId) return;
    state.slideMetadata = metadata;
    updateMagnifierOptions();
    if (metadata.cache_status === "ready") {
      $("#viewer-loading span:last-child").textContent = "正在读取本地缓存";
    }
    if (!state.viewer) {
      state.viewer = OpenSeadragon({
        id: "viewer",
        showNavigationControl: false,
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        navigatorSizeRatio: 0.16,
        animationTime: 0.45,
        blendTime: 0.08,
        maxZoomPixelRatio: 2,
        gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true, scrollToZoom: true },
      });
      state.viewer.addHandler("open", () => { $("#viewer-loading").hidden = true; $("#zoom-indicator").hidden = false; updateZoomIndicator(); renderRoiOverlays(); });
      state.viewer.addHandler("zoom", updateZoomIndicator);
      state.viewer.addHandler("animation", updateZoomIndicator);
      state.viewer.addHandler("open-failed", (event) => showViewerError(event.message || "WSI 加载失败"));
    }
    const encoded = encodeURIComponent(slide.slide_id);
    state.tileSource = {
      width: metadata.width,
      height: metadata.height,
      tileSize: metadata.tile_size,
      tileOverlap: metadata.tile_overlap,
      minLevel: 0,
      maxLevel: metadata.level_count - 1,
      getTileUrl: (level, x, y) => `/api/slides/${encoded}/tiles/${level}/${x}_${y}.jpeg`,
    };
    state.viewer.open(state.tileSource);
    if (state.magnifierEnabled) openMagnifierSource();
  } catch (error) { showViewerError(error.message); }
}

function toggleMagnifier() {
  if (state.magnifierEnabled) disableMagnifier();
  else enableMagnifier();
}

function enableMagnifier() {
  if (!state.viewer || !state.tileSource || !state.slideMetadata) {
    showToast("请先打开一张切片", true);
    return;
  }
  state.magnifierEnabled = true;
  $("#magnifier-toggle").classList.add("active");
  $("#magnifier-toggle").setAttribute("aria-pressed", "true");
  $("#magnifier-controls").hidden = false;
  updateMagnifierOptions();
  openMagnifierSource();
}

function disableMagnifier() {
  state.magnifierEnabled = false;
  $("#magnifier-toggle").classList.remove("active");
  $("#magnifier-toggle").setAttribute("aria-pressed", "false");
  $("#magnifier-controls").hidden = true;
  hideMagnifierPanel();
}

function setMagnifierTarget(target) {
  const button = $(`#magnifier-controls button[data-magnification='${target}']`);
  if (!button || button.disabled) return;
  state.magnifierTarget = target;
  updateMagnifierOptions();
  scheduleMagnifierSync();
}

function updateMagnifierOptions() {
  const objective = Number(state.slideMetadata?.objective_power) || null;
  const buttons = $$("#magnifier-controls button");
  buttons.forEach((button) => {
    const value = Number(button.dataset.magnification);
    button.disabled = Boolean(objective && value > objective + 0.1);
  });
  const selected = buttons.find((button) => Number(button.dataset.magnification) === state.magnifierTarget);
  if (selected?.disabled) {
    const available = buttons.filter((button) => !button.disabled).map((button) => Number(button.dataset.magnification));
    state.magnifierTarget = Math.max(...available);
  }
  buttons.forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.magnification) === state.magnifierTarget);
  });
  $("#magnifier-badge").textContent = `${objective ? "" : "~"}${state.magnifierTarget}×`;
}

function ensureMagnifierViewer() {
  if (state.magnifierViewer) return;
  state.magnifierViewer = OpenSeadragon({
    id: "magnifier-viewer",
    showNavigationControl: false,
    showNavigator: false,
    mouseNavEnabled: false,
    animationTime: 0,
    blendTime: 0,
    immediateRender: true,
    maxZoomPixelRatio: 1,
    visibilityRatio: 1,
    constrainDuringPan: true,
  });
  state.magnifierViewer.addHandler("open", () => {
    state.magnifierReady = true;
    scheduleMagnifierSync();
  });
  ["tile-loaded", "tile-drawing", "tile-drawn"].forEach((eventName) => {
    state.magnifierViewer.addHandler(eventName, handleMagnifierTileEvent);
  });
  state.magnifierViewer.addHandler("open-failed", () => {
    state.magnifierReady = false;
    hideMagnifierPanel();
  });
}

function handleMagnifierTileEvent(event) {
  if (event.tile && event.tile.level >= state.magnifierRequiredLevel) {
    markMagnifierRendered();
  }
}

function openMagnifierSource() {
  if (!state.magnifierEnabled || !state.tileSource || !state.selectedId) return;
  ensureMagnifierViewer();
  if (state.magnifierSlideId === state.selectedId && state.magnifierReady) return;
  state.magnifierSlideId = state.selectedId;
  state.magnifierReady = false;
  $("#magnifier-loading").hidden = false;
  state.magnifierViewer.open(state.tileSource);
}

function resetMagnifierSlide() {
  state.tileSource = null;
  state.magnifierReady = false;
  state.magnifierSlideId = null;
  state.magnifierPointer = null;
  if (state.magnifierFrame) cancelAnimationFrame(state.magnifierFrame);
  state.magnifierFrame = null;
  clearTimeout(state.magnifierLoadingTimer);
  state.magnifierLoadingVersion += 1;
  state.magnifierViewer?.close();
  hideMagnifierPanel();
}

function handleMagnifierPointerMove(event) {
  if (!state.magnifierEnabled || !state.tileSource || !state.slideMetadata) return;
  const viewerRect = $("#viewer").getBoundingClientRect();
  const point = {
    x: event.clientX - viewerRect.left,
    y: event.clientY - viewerRect.top,
  };
  if (point.x < 0 || point.y < 0 || point.x > viewerRect.width || point.y > viewerRect.height) {
    hideMagnifierPanel();
    return;
  }
  state.magnifierPointer = point;
  positionMagnifierPanel(point, viewerRect);
  $("#magnifier-panel").classList.remove("inactive");
  $("#magnifier-panel").setAttribute("aria-hidden", "false");
  scheduleMagnifierSync();
}

function positionMagnifierPanel(point, viewerRect) {
  const panel = $("#magnifier-panel");
  const offset = 18;
  const width = panel.offsetWidth;
  const height = panel.offsetHeight;
  let left = point.x + offset;
  let top = point.y + offset;
  if (left + width > viewerRect.width) left = point.x - width - offset;
  if (top + height > viewerRect.height) top = point.y - height - offset;
  left = Math.max(4, Math.min(left, viewerRect.width - width - 4));
  top = Math.max(4, Math.min(top, viewerRect.height - height - 4));
  panel.style.left = `${left}px`;
  panel.style.top = `${top}px`;
}

function scheduleMagnifierSync() {
  if (state.magnifierFrame) return;
  state.magnifierFrame = requestAnimationFrame(() => {
    state.magnifierFrame = null;
    syncMagnifier();
  });
}

function syncMagnifier() {
  if (!state.magnifierEnabled || !state.magnifierReady || !state.magnifierPointer) return;
  if (state.magnifierSlideId !== state.selectedId) return;
  const pixelPoint = new OpenSeadragon.Point(state.magnifierPointer.x, state.magnifierPointer.y);
  const viewportPoint = state.viewer.viewport.pointFromPixel(pixelPoint, true);
  const imagePoint = state.viewer.viewport.viewportToImageCoordinates(viewportPoint);
  const lensPoint = state.magnifierViewer.viewport.imageToViewportCoordinates(imagePoint);
  const objective = Number(state.slideMetadata.objective_power) || 40;
  const imageZoom = Math.min(1, state.magnifierTarget / objective);
  const viewportZoom = state.magnifierViewer.viewport.imageToViewportZoom(imageZoom);
  const maxLevel = state.slideMetadata.level_count - 1;
  state.magnifierRequiredLevel = Math.max(0, Math.floor(maxLevel + Math.log2(imageZoom)));
  markMagnifierLoading();
  state.magnifierViewer.viewport.panTo(lensPoint, true);
  state.magnifierViewer.viewport.zoomTo(viewportZoom, null, true);
  state.magnifierViewer.viewport.applyConstraints(true);
}

function markMagnifierLoading() {
  clearTimeout(state.magnifierLoadingTimer);
  const version = ++state.magnifierLoadingVersion;
  state.magnifierLoadingTimer = setTimeout(() => {
    if (version !== state.magnifierLoadingVersion) return;
    const image = state.magnifierViewer?.world?.getItemAt(0);
    if (image?.getFullyLoaded?.()) {
      $("#magnifier-loading").hidden = true;
      return;
    }
    if (state.magnifierEnabled && state.magnifierPointer) $("#magnifier-loading").hidden = false;
  }, 120);
}

function markMagnifierRendered() {
  clearTimeout(state.magnifierLoadingTimer);
  state.magnifierLoadingVersion += 1;
  $("#magnifier-loading").hidden = true;
}

function hideMagnifierPanel() {
  state.magnifierPointer = null;
  const panel = $("#magnifier-panel");
  panel.classList.add("inactive");
  panel.setAttribute("aria-hidden", "true");
}

function warmSlide(slideId) {
  fetch(`/api/slides/${encodeURIComponent(slideId)}/warm`, {
    method: "POST",
    credentials: "same-origin",
  }).catch(() => {});
}

function prefetchNextHpSlide() {
  const index = state.slides.findIndex((item) => item.slide_id === state.selectedId);
  const nextHpSlide = state.slides.slice(index + 1).find((item) => item.source === "hp");
  if (nextHpSlide) warmSlide(nextHpSlide.slide_id);
}

function updateZoomIndicator() {
  if (!state.viewer?.viewport || !state.slideMetadata) return;
  const viewportZoom = state.viewer.viewport.getZoom(true);
  const imageZoom = state.viewer.viewport.viewportToImageZoom(viewportZoom);
  const magnification = state.slideMetadata.objective_power ? imageZoom * state.slideMetadata.objective_power : null;
  $("#zoom-indicator").textContent = magnification
    ? `${magnification.toFixed(magnification < 10 ? 1 : 0)}× · ${imageZoom.toFixed(3)} px/px`
    : `${imageZoom.toFixed(3)} px/px`;
}

function toggleRoiDraw() {
  if (state.roiDrawEnabled) disableRoiDraw();
  else enableRoiDraw();
}

function enableRoiDraw() {
  if (!state.challengeSlideEligible || !state.viewer?.isOpen() || state.challengeReview?.locked) {
    showToast("当前病例不可绘制 ROI", true);
    return;
  }
  if ($("#challenge-no-roi").checked) {
    showToast("请先取消 No localizable key evidence", true);
    return;
  }
  if ((state.challengeReview?.rois || []).length >= 3) {
    showToast("每份审核最多 3 个 ROI", true);
    return;
  }
  state.roiDrawEnabled = true;
  $("#roi-draw-toggle").classList.add("active");
  $("#roi-draw-toggle").setAttribute("aria-pressed", "true");
  $("#viewer").classList.add("roi-draw-active");
}

function disableRoiDraw() {
  state.roiDrawEnabled = false;
  $("#roi-draw-toggle").classList.remove("active");
  $("#roi-draw-toggle").setAttribute("aria-pressed", "false");
  $("#viewer").classList.remove("roi-draw-active");
}

function viewerImagePoint(clientX, clientY) {
  const rect = $("#viewer").getBoundingClientRect();
  const pixel = new OpenSeadragon.Point(clientX - rect.left, clientY - rect.top);
  const viewport = state.viewer.viewport.pointFromPixel(pixel, true);
  return state.viewer.viewport.viewportToImageCoordinates(viewport);
}

function startRoiDrawing(event) {
  if (!state.roiDrawEnabled || event.button !== 0 || event.target.closest("#magnifier-panel")) return;
  event.preventDefault();
  event.stopPropagation();
  const viewerRect = $("#viewer").getBoundingClientRect();
  const start = { x: event.clientX - viewerRect.left, y: event.clientY - viewerRect.top };
  const box = document.createElement("div");
  box.className = "roi-drawing-box";
  box.style.left = `${start.x}px`;
  box.style.top = `${start.y}px`;
  $("#viewer").append(box);
  state.viewer.setMouseNavEnabled(false);

  const move = (moveEvent) => {
    const x = Math.max(0, Math.min(viewerRect.width, moveEvent.clientX - viewerRect.left));
    const y = Math.max(0, Math.min(viewerRect.height, moveEvent.clientY - viewerRect.top));
    box.style.left = `${Math.min(start.x, x)}px`;
    box.style.top = `${Math.min(start.y, y)}px`;
    box.style.width = `${Math.abs(x - start.x)}px`;
    box.style.height = `${Math.abs(y - start.y)}px`;
  };
  const finish = async (upEvent) => {
    document.removeEventListener("pointermove", move, true);
    document.removeEventListener("pointerup", finish, true);
    box.remove();
    state.viewer.setMouseNavEnabled(true);
    const endX = Math.max(viewerRect.left, Math.min(viewerRect.right, upEvent.clientX));
    const endY = Math.max(viewerRect.top, Math.min(viewerRect.bottom, upEvent.clientY));
    if (Math.abs(endX - event.clientX) < 8 || Math.abs(endY - event.clientY) < 8) return;
    const first = viewerImagePoint(event.clientX, event.clientY);
    const second = viewerImagePoint(endX, endY);
    const geometry = normalizedRoiGeometry(first.x, first.y, second.x, second.y);
    await createRoi(geometry);
  };
  document.addEventListener("pointermove", move, true);
  document.addEventListener("pointerup", finish, true);
}

function normalizedRoiGeometry(x1, y1, x2, y2) {
  const width = state.slideMetadata.width;
  const height = state.slideMetadata.height;
  const left = Math.max(0, Math.min(width, Math.min(x1, x2)));
  const top = Math.max(0, Math.min(height, Math.min(y1, y2)));
  const right = Math.max(0, Math.min(width, Math.max(x1, x2)));
  const bottom = Math.max(0, Math.min(height, Math.max(y1, y2)));
  return { x_level0: left, y_level0: top, width_level0: right - left, height_level0: bottom - top };
}

function roiApiPayload(roi, reviewVersion = state.challengeReview.version) {
  const mppX = Number(state.slideMetadata?.mpp_x) || null;
  const mppY = Number(state.slideMetadata?.mpp_y) || mppX;
  return {
    review_version: reviewVersion,
    version: roi.version || 0,
    x_level0: roi.x_level0, y_level0: roi.y_level0,
    width_level0: roi.width_level0, height_level0: roi.height_level0,
    mpp_x: mppX, mpp_y: mppY,
    physical_width_um: mppX ? roi.width_level0 * mppX : null,
    physical_height_um: mppY ? roi.height_level0 * mppY : null,
    viewer_zoom: state.viewer?.viewport?.getZoom(true) || null,
    diagnostic_role: roi.diagnostic_role || "",
    differential_direction: roi.differential_direction || "",
    evidence_strength: roi.evidence_strength || "",
    evidence: roi.evidence || [],
    note: roi.note || "",
  };
}

async function createRoi(geometry) {
  try {
    await flushChallengeDraft();
    await state.roiSaveQueue.catch(() => {});
    const review = await ensureChallengeReview();
    const response = await api(`/api/challenge-reviews/${review.id}/rois`, {
      method: "POST", body: JSON.stringify(roiApiPayload({ ...geometry, version: 0 }, review.version)),
    });
    state.challengeReview.version = response.review_version;
    state.challengeReview.rois.push(response.roi);
    state.selectedRoiId = response.roi.id;
    fillChallengeReview(state.challengeReview);
    renderRoiOverlays();
    if (state.challengeReview.rois.length >= 3) disableRoiDraw();
  } catch (error) { showToast(error.message, true); }
}

function clearRoiOverlays() {
  state.roiOverlays.forEach(({ element }) => {
    try { state.viewer?.removeOverlay(element); } catch (_) { element.remove(); }
  });
  state.roiOverlays.clear();
  state.selectedRoiId = null;
  disableRoiDraw();
}

function renderRoiOverlays() {
  if (!state.viewer?.isOpen()) return;
  state.roiOverlays.forEach(({ element }) => state.viewer.removeOverlay(element));
  state.roiOverlays.clear();
  (state.challengeReview?.rois || []).forEach((roi) => {
    const element = document.createElement("div");
    element.className = `roi-overlay${state.selectedRoiId === roi.id ? " selected" : ""}`;
    element.dataset.roiId = roi.id;
    element.innerHTML = `<span class="roi-number">ROI ${roi.display_order}</span><span class="roi-handle" title="调整大小"></span>`;
    element.addEventListener("pointerdown", (event) => startRoiTransform(event, roi, event.target.classList.contains("roi-handle") ? "resize" : "move"));
    element.addEventListener("click", (event) => { event.stopPropagation(); selectRoi(roi.id); });
    const rect = state.viewer.viewport.imageToViewportRectangle(roi.x_level0, roi.y_level0, roi.width_level0, roi.height_level0);
    state.viewer.addOverlay({ element, location: rect });
    state.roiOverlays.set(roi.id, { element });
  });
}

function selectRoi(roiId) {
  state.selectedRoiId = roiId;
  renderRoiOverlays();
  renderRoiEditors();
  $(`.roi-editor[data-roi-id='${roiId}']`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function startRoiTransform(event, roi, mode) {
  if (state.challengeReview?.locked || event.button !== 0) return;
  event.preventDefault();
  event.stopPropagation();
  selectRoi(roi.id);
  const startPoint = viewerImagePoint(event.clientX, event.clientY);
  const original = { ...roi };
  state.viewer.setMouseNavEnabled(false);
  const move = (moveEvent) => {
    const point = viewerImagePoint(moveEvent.clientX, moveEvent.clientY);
    if (mode === "move") {
      roi.x_level0 = Math.max(0, Math.min(state.slideMetadata.width - roi.width_level0, original.x_level0 + point.x - startPoint.x));
      roi.y_level0 = Math.max(0, Math.min(state.slideMetadata.height - roi.height_level0, original.y_level0 + point.y - startPoint.y));
    } else {
      roi.width_level0 = Math.max(20, Math.min(state.slideMetadata.width - original.x_level0, original.width_level0 + point.x - startPoint.x));
      roi.height_level0 = Math.max(20, Math.min(state.slideMetadata.height - original.y_level0, original.height_level0 + point.y - startPoint.y));
    }
    const overlay = state.roiOverlays.get(roi.id)?.element;
    if (overlay) {
      const rect = state.viewer.viewport.imageToViewportRectangle(roi.x_level0, roi.y_level0, roi.width_level0, roi.height_level0);
      state.viewer.updateOverlay(overlay, rect);
    }
  };
  const finish = async () => {
    document.removeEventListener("pointermove", move, true);
    document.removeEventListener("pointerup", finish, true);
    state.viewer.setMouseNavEnabled(true);
    await updateRoi(roi, false);
  };
  document.addEventListener("pointermove", move, true);
  document.addEventListener("pointerup", finish, true);
}

function evidenceCodesForChallenge() {
  const taxonomy = state.challengeTaxonomy;
  return [...new Set([...(taxonomy.evidence_by_challenge[$("#challenge-primary").value] || []), ...taxonomy.general_evidence])];
}

function directionCodesForChallenge() {
  return state.challengeTaxonomy.directions_by_challenge[$("#challenge-primary").value] || [];
}

function renderRoiEditors() {
  const rois = state.challengeReview?.rois || [];
  $("#roi-count").textContent = `${rois.length} / 3`;
  $("#roi-empty").hidden = rois.length > 0;
  const container = $("#roi-editors");
  container.replaceChildren();
  rois.forEach((roi) => {
    const editor = document.createElement("article");
    editor.className = `roi-editor${state.selectedRoiId === roi.id ? " selected" : ""}`;
    editor.dataset.roiId = roi.id;
    const fov = roi.physical_width_um && roi.physical_height_um
      ? `${(roi.physical_width_um / 1000).toFixed(2)} × ${(roi.physical_height_um / 1000).toFixed(2)} mm`
      : `${Math.round(roi.width_level0)} × ${Math.round(roi.height_level0)} px`;
    const selectedEvidence = new Map((roi.evidence || []).map((item) => [item.evidence_code, item.evidence_other]));
    editor.innerHTML = `
      <div class="roi-editor-heading"><strong>ROI ${roi.display_order}</strong><span>FOV ${fov}</span><button type="button" data-action="up" title="上移"${roi.display_order === 1 ? " disabled" : ""}><i data-lucide="arrow-up"></i></button><button type="button" data-action="down" title="下移"${roi.display_order === rois.length ? " disabled" : ""}><i data-lucide="arrow-down"></i></button><button type="button" data-action="delete" title="删除 ROI"><i data-lucide="trash-2"></i></button></div>
      <div class="roi-editor-body">
        <label>Evidence type<div class="evidence-options">${evidenceCodesForChallenge().map((code) => `<label><input type="checkbox" data-field="evidence" value="${code}"${selectedEvidence.has(code) ? " checked" : ""}><span>${challengeLabel(code)}</span></label>`).join("")}</div></label>
        <label class="roi-other-evidence"${selectedEvidence.has("other") ? "" : " hidden"}>Other evidence<input data-field="evidence_other" maxlength="500" value="${escapeHtml(selectedEvidence.get("other") || "")}"></label>
        <div class="field-pair"><label>Diagnostic role<select data-field="diagnostic_role"><option value="">请选择</option>${state.challengeTaxonomy.diagnostic_roles.map((code) => `<option value="${code}"${roi.diagnostic_role === code ? " selected" : ""}>${challengeLabel(code)}</option>`).join("")}</select></label><label>Strength<select data-field="evidence_strength"><option value="">请选择</option>${state.challengeTaxonomy.evidence_strengths.map((code) => `<option value="${code}"${roi.evidence_strength === code ? " selected" : ""}>${challengeLabel(code)}</option>`).join("")}</select></label></div>
        <label>Differential direction<select data-field="differential_direction"><option value="">请选择</option>${directionCodesForChallenge().map((code) => `<option value="${code}"${roi.differential_direction === code ? " selected" : ""}>${challengeLabel(code)}</option>`).join("")}</select></label>
        <label>ROI note<textarea data-field="note" rows="2" maxlength="5000">${escapeHtml(roi.note || "")}</textarea></label>
      </div>`;
    editor.addEventListener("click", () => {
      if (state.selectedRoiId !== roi.id) selectRoi(roi.id);
    });
    editor.querySelector("[data-action='up']").addEventListener("click", (event) => { event.stopPropagation(); reorderRoi(roi.id, -1); });
    editor.querySelector("[data-action='down']").addEventListener("click", (event) => { event.stopPropagation(); reorderRoi(roi.id, 1); });
    editor.querySelector("[data-action='delete']").addEventListener("click", (event) => { event.stopPropagation(); deleteRoi(roi.id); });
    editor.querySelectorAll("input,select,textarea").forEach((control) => {
      control.disabled = Boolean(state.challengeReview.locked);
      control.addEventListener("change", () => scheduleRoiEditorSave(roi, editor));
      if (control.tagName === "TEXTAREA" || control.dataset.field === "evidence_other") {
        control.addEventListener("input", () => scheduleRoiEditorSave(roi, editor));
      }
    });
    container.append(editor);
  });
  lucide.createIcons();
}

function updateRoiFromEditor(roi, editor) {
  const otherChecked = editor.querySelector(`input[data-field='evidence'][value='other']`)?.checked;
  editor.querySelector(".roi-other-evidence").hidden = !otherChecked;
  roi.diagnostic_role = editor.querySelector(`[data-field='diagnostic_role']`).value;
  roi.differential_direction = editor.querySelector(`[data-field='differential_direction']`).value;
  roi.evidence_strength = editor.querySelector(`[data-field='evidence_strength']`).value;
  roi.note = editor.querySelector(`[data-field='note']`).value;
  roi.evidence = [...editor.querySelectorAll(`input[data-field='evidence']:checked`)].map((input) => ({
    evidence_code: input.value,
    evidence_other: input.value === "other" ? editor.querySelector(`[data-field='evidence_other']`).value : "",
  }));
  updateRoi(roi, false);
}

function scheduleRoiEditorSave(roi, editor) {
  clearTimeout(roi.saveTimer);
  roi.saveTimer = setTimeout(() => updateRoiFromEditor(roi, editor), 800);
}

async function updateRoi(roi, rerender = true) {
  const operation = async () => {
    try {
      const response = await api(`/api/challenge-rois/${roi.id}`, {
        method: "PUT", body: JSON.stringify(roiApiPayload(roi)),
      });
      state.challengeReview.version = response.review_version;
      roi.version = response.roi.version;
      roi.physical_width_um = response.roi.physical_width_um;
      roi.physical_height_um = response.roi.physical_height_um;
      $("#challenge-save-state").textContent = `${state.challengeReview.status === "submitted" ? "已提交" : "草稿"} · v${state.challengeReview.version}`;
      if (rerender) { renderRoiEditors(); renderRoiOverlays(); }
      return response;
    } catch (error) {
      showToast(error.message, true);
      await reloadChallengeReview();
      throw error;
    }
  };
  state.roiSaveQueue = state.roiSaveQueue.catch(() => {}).then(operation);
  return state.roiSaveQueue;
}

async function deleteRoi(roiId) {
  try {
    await state.roiSaveQueue.catch(() => {});
    const response = await api(`/api/challenge-rois/${roiId}`, {
      method: "DELETE", body: JSON.stringify({ version: state.challengeReview.version }),
    });
    state.challengeReview.version = response.review_version;
    state.challengeReview.rois = state.challengeReview.rois.filter((roi) => roi.id !== roiId);
    state.challengeReview.rois.forEach((roi, index) => { roi.display_order = index + 1; });
    state.selectedRoiId = null;
    renderRoiEditors();
    renderRoiOverlays();
  } catch (error) { showToast(error.message, true); }
}

async function reorderRoi(roiId, delta) {
  const rois = state.challengeReview?.rois || [];
  const index = rois.findIndex((roi) => roi.id === roiId);
  const target = index + delta;
  if (index < 0 || target < 0 || target >= rois.length) return;
  try {
    await state.roiSaveQueue.catch(() => {});
    const ordered = [...rois];
    [ordered[index], ordered[target]] = [ordered[target], ordered[index]];
    const review = await api(`/api/challenge-reviews/${state.challengeReview.id}/rois/order`, {
      method: "PUT",
      body: JSON.stringify({ version: state.challengeReview.version, ordered_roi_ids: ordered.map((roi) => roi.id) }),
    });
    state.challengeReview = review;
    state.selectedRoiId = roiId;
    fillChallengeReview(review);
    renderRoiOverlays();
  } catch (error) { showToast(error.message, true); await reloadChallengeReview(); }
}

function showViewerError(message) {
  $("#viewer-loading").hidden = true;
  const error = $("#viewer-error");
  error.querySelector("span").textContent = message;
  error.hidden = false;
}

async function saveReview(event) {
  event.preventDefault();
  if (!state.selectedId) return;
  const payload = {
    revised_label: $("#review-label").value,
    status: $("input[name='review-status']:checked").value,
    diagnostic_confidence: $("input[name='review-confidence']:checked").value,
    quality_flags: $$("input[name='quality-flag']:checked").map((input) => input.value),
    notes: $("#review-notes").value,
  };
  try {
    await api(`/api/slides/${encodeURIComponent(state.selectedId)}/review`, { method: "PUT", body: JSON.stringify(payload) });
    showToast("审核已保存");
    await loadProgress();
    const index = state.slides.findIndex((item) => item.slide_id === state.selectedId);
    const nextId = state.slides[index + 1]?.slide_id;
    await loadSlides(state.selectedId);
    if (nextId) await selectSlide(nextId);
  } catch (error) { showToast(error.message, true); }
}

async function loadProgress() {
  try {
    const data = await api("/api/review-progress");
    $("#progress-text").textContent = `提交 ${data.challenge_submitted.toLocaleString()} / ${data.core_total.toLocaleString()} · 保留 ${data.retained_challenges} · 待仲裁 ${data.pending_adjudication} · 可排除 ${data.eligible_for_exclusion}`;
    $("#progress-bar").style.width = `${data.core_total ? (100 * data.challenge_submitted / data.core_total) : 0}%`;
  } catch (_) {}
}

function moveSelection(delta) {
  if (!state.slides.length) return;
  const current = state.slides.findIndex((item) => item.slide_id === state.selectedId);
  const target = state.slides[current + delta];
  if (target) selectSlide(target.slide_id);
  else if (delta > 0 && state.page < state.pages) { state.page += 1; loadSlides(); }
  else if (delta < 0 && state.page > 1) { state.page -= 1; loadSlides(); }
}

function changePage(delta) {
  const target = state.page + delta;
  if (target < 1 || target > state.pages) return;
  state.page = target;
  state.selectedId = null;
  loadSlides();
}

function clearFilters() {
  $("#filter-search").value = "";
  $("#filter-core").checked = false;
  ["#filter-label", "#filter-consensus", "#filter-source", "#filter-fold", "#filter-status", "#filter-consistency", "#filter-resolution"].forEach((selector) => { $(selector).value = ""; });
  $("#filter-sort").value = "consensus_desc";
  state.page = 1;
  loadSlides();
}

function clearSelection() {
  state.wsiRequestId += 1;
  resetMagnifierSlide();
  clearRoiOverlays();
  state.selectedId = null;
  $("#no-selection").hidden = false;
  $("#review-content").hidden = true;
  $("#viewer-empty").hidden = false;
  $("#viewer-error").hidden = true;
  $("#viewer-loading").hidden = true;
  state.viewer?.close();
}

async function openAdmin() {
  $("#admin-dialog").showModal();
  await loadUsers();
  lucide.createIcons();
}

async function loadUsers() {
  try {
    const users = await api("/api/admin/users");
    const body = $("#user-body");
    body.replaceChildren();
    users.forEach((user) => {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${escapeHtml(user.username)}</td><td>${escapeHtml(user.display_name)}</td><td>${user.role}</td><td>${user.active ? "启用" : "停用"}</td><td><div class="user-actions"><button data-action="password" title="重置密码"><i data-lucide="key-round"></i></button><button data-action="active" title="${user.active ? "停用" : "启用"}"><i data-lucide="${user.active ? "user-x" : "user-check"}"></i></button></div></td>`;
      row.querySelector("[data-action='password']").addEventListener("click", () => resetUserPassword(user));
      row.querySelector("[data-action='active']").addEventListener("click", () => toggleUserActive(user));
      body.append(row);
    });
    lucide.createIcons();
  } catch (error) { showToast(error.message, true); }
}

async function createUser(event) {
  event.preventDefault();
  const payload = {
    username: $("#new-username").value,
    display_name: $("#new-display-name").value,
    password: $("#new-password").value,
    role: $("#new-role").value,
  };
  try {
    await api("/api/admin/users", { method: "POST", body: JSON.stringify(payload) });
    event.target.reset();
    showToast("用户已创建");
    await loadUsers();
  } catch (error) { showToast(error.message, true); }
}

async function resetUserPassword(user) {
  const password = window.prompt(`为 ${user.display_name} 设置新密码（至少 10 位）`);
  if (!password) return;
  try {
    await api(`/api/admin/users/${user.id}/password`, { method: "PUT", body: JSON.stringify({ password }) });
    showToast("密码已重置");
  } catch (error) { showToast(error.message, true); }
}

async function toggleUserActive(user) {
  if (!window.confirm(`${user.active ? "停用" : "启用"}用户 ${user.display_name}？`)) return;
  try {
    await api(`/api/admin/users/${user.id}/active?active=${user.active ? "false" : "true"}`, { method: "PUT" });
    showToast(user.active ? "用户已停用" : "用户已启用");
    await loadUsers();
  } catch (error) { showToast(error.message, true); }
}

function statusText(statusValue) {
  return { unreviewed: "未审核", completed: "已完成", questionable: "存疑" }[statusValue] || statusValue;
}

function formatTime(value) {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

initialize().catch((error) => { $("#login-error").textContent = error.message; });
