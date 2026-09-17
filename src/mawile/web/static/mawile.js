(function () {
  const form = document.getElementById("config-form");
  focusFirstError();
  if (!form || !window.FormData || !window.XMLHttpRequest) {
    return;
  }

  const actionButtons = Array.from(form.querySelectorAll(".summary-actions button[formaction]"));
  const autosaveStatus = document.querySelector("[data-autosave-status]");
  let autosaveTimer = null;
  let autosaveRequestId = 0;
  let submittingConfigForm = false;

  actionButtons.forEach(function (button) {
    button.addEventListener("click", function (event) {
      event.preventDefault();
      submitConfigForm(form, button);
    });
  });
  setupAutosave();

  async function submitConfigForm(form, button) {
    if (typeof form.checkValidity === "function" && !form.checkValidity()) {
      // A constraint may be violated inside a collapsed <details>, which the
      // browser cannot focus — reveal it so the user sees why, then stop.
      const invalid = form.querySelector(":invalid");
      if (invalid) {
        let section = invalid.closest("details:not([open])");
        while (section) {
          section.open = true;
          section = section.parentElement && section.parentElement.closest("details:not([open])");
        }
        if (typeof invalid.reportValidity === "function") {
          invalid.reportValidity();
        }
      }
      return;
    }

    const action = button.getAttribute("formaction") || form.getAttribute("action") || window.location.pathname;
    const method = (button.getAttribute("formmethod") || form.getAttribute("method") || "post").toUpperCase();
    const originalHtml = button.innerHTML;
    const busyLabel = action === "/runs/start" ? "Starting..." : "Suggesting...";

    submittingConfigForm = true;
    actionButtons.forEach(function (item) {
      item.disabled = true;
    });
    button.textContent = busyLabel;

    try {
      const response = await postForm(action, method, new FormData(form));
      if (response.ok && response.url) {
        window.location.href = response.url;
        return;
      }
      replaceDocument(response.html);
    } catch (error) {
      actionButtons.forEach(function (item) {
        item.disabled = false;
      });
      button.innerHTML = originalHtml;
      submittingConfigForm = false;
      const message = document.createElement("div");
      message.className = "alert error";
      message.tabIndex = -1;
      message.setAttribute("data-autofocus-error", "");
      message.innerHTML = "<strong>Could not submit</strong><p>The audit form could not be sent. Check that the local server is still running.</p>";
      form.before(message);
      focusFirstError();
    }
  }

  function setupAutosave() {
    Array.from(form.querySelectorAll("input, select, textarea")).forEach(function (field) {
      field.addEventListener("input", scheduleAutosave);
      field.addEventListener("change", scheduleAutosave);
    });
  }

  function scheduleAutosave() {
    if (submittingConfigForm) {
      return;
    }
    window.clearTimeout(autosaveTimer);
    if (autosaveStatus) {
      autosaveStatus.textContent = "Saving changes for this session...";
    }
    autosaveTimer = window.setTimeout(runAutosave, 700);
  }

  function runAutosave() {
    if (submittingConfigForm) {
      return;
    }
    const requestId = autosaveRequestId + 1;
    autosaveRequestId = requestId;
    postJson("/configure/autosave", "POST", new FormData(form)).then(function (result) {
      if (requestId !== autosaveRequestId || !autosaveStatus) {
        return;
      }
      autosaveStatus.textContent = result.ok
        ? "Changes saved for this session."
        : "Some fields need attention before changes can save.";
    }).catch(function () {
      if (requestId === autosaveRequestId && autosaveStatus) {
        autosaveStatus.textContent = "Could not autosave. Run Audit and Suggest still use the current form.";
      }
    });
  }

  function postForm(action, method, data) {
    return new Promise(function (resolve, reject) {
      const request = new XMLHttpRequest();
      request.open(method, action, true);
      request.onload = function () {
        resolve({
          html: request.responseText,
          ok: request.status >= 200 && request.status < 400,
          status: request.status,
          url: request.responseURL
        });
      };
      request.onerror = reject;
      request.send(data);
    });
  }

  function postJson(action, method, data) {
    return new Promise(function (resolve, reject) {
      const request = new XMLHttpRequest();
      request.open(method, action, true);
      request.onload = function () {
        let payload = {};
        try {
          payload = JSON.parse(request.responseText || "{}");
        } catch (error) {
          payload = {};
        }
        resolve({
          ok: request.status >= 200 && request.status < 400 && payload.ok !== false,
          payload: payload
        });
      };
      request.onerror = reject;
      request.send(data);
    });
  }

  function replaceDocument(html) {
    document.open();
    document.write(html);
    document.close();
    window.setTimeout(focusFirstError, 0);
  }

  function focusFirstError() {
    const target = document.querySelector("[data-autofocus-error], .alert.error");
    if (!target) {
      return;
    }
    target.scrollIntoView({ block: "start", behavior: "smooth" });
    if (typeof target.focus === "function") {
      target.focus({ preventScroll: true });
    }
  }
})();

(function () {
  const table = document.querySelector("[data-invariant-table]");
  const empty = document.querySelector("[data-suggested-empty]");
  const radios = Array.from(document.querySelectorAll('input[name="operator_mode"]'));
  const invariantBoxes = Array.from(document.querySelectorAll('input[name="perturbation_operators"]'));
  const directionalBoxes = Array.from(document.querySelectorAll('input[name="directional_operators"]'));
  const invariantCount = document.querySelector("[data-invariant-count]");
  const directionalCount = document.querySelector("[data-directional-count]");
  const customCount = document.querySelector("[data-custom-count]");
  const customRows = Array.from(document.querySelectorAll("[data-custom-row]"));
  const yamlBlock = document.querySelector("[data-config-yaml]");
  const operatorBody = document.querySelector("[data-operator-body]");
  const operatorBoxes = Array.from(document.querySelectorAll("[data-operator-row]"));
  const runnableCount = document.querySelector("[data-runnable-count]");
  const summaryChecks = document.querySelector("[data-summary-checks]");
  const previewBody = document.querySelector("[data-preview-body]");
  const previewCount = document.querySelector("[data-preview-count]");
  const knownOperators = new Set(operatorBoxes.map(function (box) {
    return box.value;
  }));
  const staticOperatorRows = operatorBody
    ? Array.from(operatorBody.querySelectorAll("tr")).filter(function (row) {
      const operator = row.querySelector("td:nth-child(2) code");
      return operator && !knownOperators.has(operator.textContent);
    }).map(function (row) {
      return row.outerHTML;
    })
    : [];
  const staticPreviewCards = previewBody
    ? Array.from(previewBody.querySelectorAll("[data-preview-card]")).filter(function (card) {
      return !knownOperators.has(card.getAttribute("data-preview-operator") || "");
    }).map(function (card) {
      return card.outerHTML;
    })
    : [];
  if (!table || !radios.length) {
    return;
  }

  function syncInvariantMode() {
    const selected = radios.find(function (radio) {
      return radio.checked;
    });
    const showEmpty = Boolean(empty && selected && selected.value === "suggest");
    if (empty) {
      empty.hidden = !showEmpty;
    }
    table.hidden = showEmpty;
  }

  function syncOperatorSelection(updateYaml) {
    const invariantSelected = checkedValues(invariantBoxes);
    const directionalSelected = checkedValues(directionalBoxes);
    if (invariantCount) {
      invariantCount.textContent = String(invariantSelected.length);
    }
    if (directionalCount) {
      directionalCount.textContent = String(directionalSelected.length);
    }
    syncCustomSelection();
    if (updateYaml !== false) {
      updateYamlPreview(invariantSelected, directionalSelected);
    }
    syncOperatorPreview();
  }

  function checkedValues(boxes) {
    return boxes.filter(function (box) {
      return box.checked;
    }).map(function (box) {
      return box.value;
    });
  }

  function syncCustomSelection() {
    if (!customCount) {
      return;
    }
    const selected = customRows.filter(function (row) {
      const enabled = row.querySelector("[data-custom-enabled]");
      const name = row.querySelector("[data-custom-name]");
      const instruction = row.querySelector("[data-custom-instruction]");
      const hasContent = Boolean(
        (name && name.value.trim()) || (instruction && instruction.value.trim())
      );
      return Boolean(enabled && enabled.checked && hasContent);
    });
    customCount.textContent = String(selected.length);
  }

  function updateYamlPreview(invariantSelected, directionalSelected) {
    if (!yamlBlock) {
      return;
    }
    const invariantYaml = invariantSelected.length === invariantBoxes.length ? [] : invariantSelected;
    yamlBlock.textContent = updateAuditOperatorYaml(
      yamlBlock.textContent,
      invariantYaml,
      directionalSelected
    );
  }

  function updateAuditOperatorYaml(text, invariantValues, directionalValues) {
    const lines = String(text || "").split("\n");
    const auditIndex = lines.findIndex(function (line) {
      return line === "audit:";
    });
    if (auditIndex === -1) {
      return text;
    }
    let end = lines.length;
    for (let index = auditIndex + 1; index < lines.length; index += 1) {
      if (lines[index] && !lines[index].startsWith(" ")) {
        end = index;
        break;
      }
    }

    const body = [];
    for (let index = auditIndex + 1; index < end; index += 1) {
      const line = lines[index];
      if (isOperatorKey(line)) {
        index = skipYamlList(lines, index + 1, end) - 1;
        continue;
      }
      body.push(line);
    }

    const inserted = [
      ...yamlListLines("perturbation_operators", invariantValues),
      ...yamlListLines("directional_perturbation_operators", directionalValues)
    ];
    return [
      ...lines.slice(0, auditIndex + 1),
      ...inserted,
      ...body,
      ...lines.slice(end)
    ].join("\n");
  }

  function isOperatorKey(line) {
    return /^  (perturbation_operators|directional_perturbation_operators):/.test(line);
  }

  function skipYamlList(lines, start, end) {
    let index = start;
    while (index < end && (/^  - /.test(lines[index]) || /^    /.test(lines[index]))) {
      index += 1;
    }
    return index;
  }

  function yamlListLines(key, values) {
    if (!values.length) {
      return [`  ${key}: []`];
    }
    return [`  ${key}:`].concat(values.map(function (value) {
      return `  - ${value}`;
    }));
  }

  function syncOperatorPreview() {
    if (!operatorBody) {
      return;
    }
    const selected = operatorBoxes.filter(function (box) {
      return box.checked;
    });
    const total = selected.length + staticOperatorRows.length;
    if (runnableCount) {
      runnableCount.textContent = String(total);
    }
    if (summaryChecks) {
      summaryChecks.textContent = String(total);
    }
    if (!total) {
      operatorBody.innerHTML = '<tr><td colspan="7"><span class="muted">No perturbations selected yet.</span></td></tr>';
      syncPreviewCards(selected);
      return;
    }
    operatorBody.innerHTML = selected.map(operatorRowHtml).concat(staticOperatorRows).join("");
    syncPreviewCards(selected);
  }

  function operatorRowHtml(box) {
    const data = box.dataset;
    const touches = data.touches
      ? `<br><span class="muted">${escapeHtml(data.touches)}</span>`
      : "";
    const validation = data.operator === "judge_rubric_criterion_reorder"
      ? "Structural guard or LLM validation"
      : data.validationKind
        ? `LLM validator: ${escapeHtml(data.validationKind)}`
        : '<span class="muted">Structural guard</span>';
    const statusOk = data.statusOk === "true";
    const statusClass = statusOk ? "ok" : "bad";
    const statusGlyph = statusOk ? "&#10003;" : "&times;";
    return [
      "<tr>",
      `<td><span class="pill t-${escapeHtml(data.familyAccent || "default")}">${escapeHtml(data.familyLabel || "")}</span></td>`,
      `<td><strong>${escapeHtml(data.dimension || "")}</strong><br><code>${escapeHtml(data.operator || box.value)}</code></td>`,
      `<td>${escapeHtml(data.type || "")}</td>`,
      `<td><code>${escapeHtml(data.target || "")}</code>${touches}</td>`,
      `<td>${escapeHtml(data.expected || "")}</td>`,
      `<td>${validation}</td>`,
      `<td><span class="status-glyph ${statusClass}" title="${escapeHtml(data.status || "")}" aria-label="${escapeHtml(data.status || "")}">${statusGlyph}</span></td>`,
      "</tr>"
    ].join("");
  }

  function syncPreviewCards(selected) {
    if (!previewBody) {
      return;
    }
    const cards = selected.filter(function (box) {
      return box.dataset.previewSummary;
    }).map(previewCardHtml).concat(staticPreviewCards);
    if (previewCount) {
      previewCount.textContent = String(cards.length);
    }
    previewBody.innerHTML = cards.length
      ? cards.join("")
      : '<p class="muted">No preview examples are available yet.</p>';
  }

  function previewCardHtml(box) {
    const data = box.dataset;
    return [
      `<div class="preview-card" data-preview-card data-preview-operator="${escapeHtml(data.operator || box.value)}">`,
      `<div class="preview-top"><span class="pill t-${escapeHtml(data.familyAccent || "default")}">${escapeHtml(data.familyLabel || "")}</span><strong>${escapeHtml(data.dimension || "")}</strong></div>`,
      `<p>${escapeHtml(data.previewSummary || "")}</p>`,
      '<div class="diff-mini">',
      `<div><span>Before</span><pre>${escapeHtml(data.previewBefore || "")}</pre></div>`,
      `<div><span>After</span><pre>${escapeHtml(data.previewAfter || "Generated during the run.")}</pre></div>`,
      '</div>',
      '</div>'
    ].join("");
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (character) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[character];
    });
  }

  radios.forEach(function (radio) {
    radio.addEventListener("change", syncInvariantMode);
  });
  invariantBoxes.concat(directionalBoxes).forEach(function (box) {
    box.addEventListener("change", function () {
      syncOperatorSelection(true);
    });
  });
  customRows.forEach(function (row) {
    Array.from(row.querySelectorAll("[data-custom-enabled], [data-custom-name], [data-custom-instruction]")).forEach(function (field) {
      field.addEventListener("input", syncCustomSelection);
      field.addEventListener("change", syncCustomSelection);
    });
  });
  syncInvariantMode();
  syncOperatorSelection(false);
})();

(function () {
  const tabs = Array.from(document.querySelectorAll(".tabs a[data-tab]"));
  const panels = Array.from(document.querySelectorAll(".tab-panel"));
  if (!tabs.length || !panels.length) {
    return;
  }

  function activate(tabId) {
    let matched = false;
    panels.forEach(function (panel) {
      const isActive = panel.id === tabId;
      panel.hidden = !isActive;
      matched = matched || isActive;
    });
    tabs.forEach(function (tab) {
      tab.classList.toggle("active", tab.getAttribute("data-tab") === tabId);
    });
    return matched;
  }

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function (event) {
      event.preventDefault();
      activate(tab.getAttribute("data-tab"));
      history.replaceState(null, "", "#" + tab.getAttribute("data-tab"));
    });
  });

  const initial = window.location.hash.replace("#", "");
  if (!initial || !activate(initial)) {
    activate(tabs[0].getAttribute("data-tab"));
  }
})();

(function () {
  const progress = document.querySelector("[data-job-id]");
  if (!progress || !window.EventSource) {
    return;
  }

  const jobId = progress.getAttribute("data-job-id");
  const text = progress.querySelector("[data-progress-text]");
  const bar = progress.querySelector("[data-progress-bar]");
  const source = new EventSource(`/runs/events/${jobId}`);

  source.onmessage = function (message) {
    const event = JSON.parse(message.data);
    const percent = progressPercent(event);
    if (text) {
      text.textContent = progressText(event);
    }
    if (bar) {
      bar.style.width = `${percent}%`;
    }
    if (event.stage === "complete" && event.run_id) {
      source.close();
      window.setTimeout(function () {
        window.location.href = `/report?run=${encodeURIComponent(event.run_id)}`;
      }, 500);
    }
    if (event.stage === "error") {
      source.close();
    }
  };

  source.onerror = function () {
    if (text) {
      text.textContent = "Progress connection interrupted.";
    }
    source.close();
  };

  function progressText(event) {
    const message = event.message || titleCase(event.stage || "running");
    if (event.total && event.current !== undefined) {
      return `${message}: ${event.current}/${event.total}`;
    }
    return message;
  }

  function progressPercent(event) {
    const stage = event.stage || "";
    const total = Number(event.total || 0);
    const current = Number(event.current || 0);
    if (stage === "baseline_judge_calls_started" || stage === "baseline_judge_call_completed" || stage === "baseline_judge_calls_complete") {
      if (total > 0) {
        return Math.min(23, Math.max(12, 12 + Math.floor((current / total) * 11)));
      }
      return 12;
    }
    if (stage === "generation_unit_completed") {
      if (total > 0) {
        return Math.min(40, Math.max(28, 28 + Math.floor((current / total) * 12)));
      }
      return 28;
    }
    if (stage === "validation_call_completed") {
      if (total > 0) {
        return Math.min(45, Math.max(40, 40 + Math.floor((current / total) * 5)));
      }
      return 40;
    }
    if (stage === "judge_calls_started" || stage === "judge_call_completed" || stage === "judge_calls_complete") {
      if (total > 0) {
        return Math.min(85, Math.max(45, 45 + Math.floor((current / total) * 40)));
      }
      return 45;
    }
    const map = {
      started: 2,
      planning_complete: 8,
      loading_items: 12,
      building_perturbations: 24,
      generating_perturbations: 28,
      validating_perturbations: 40,
      computing_metrics: 90,
      writing_artifacts: 96,
      complete: 100,
      error: 100
    };
    return map[stage] || 6;
  }

  function titleCase(value) {
    return String(value).replace(/[_-]+/g, " ").replace(/\b\w/g, function (letter) {
      return letter.toUpperCase();
    });
  }
})();

(function () {
  Array.from(document.querySelectorAll("[data-submit-on-change]")).forEach(function (input) {
    input.addEventListener("change", function () {
      if (input.files && input.files.length) {
        input.form.submit();
      }
    });
  });
})();
