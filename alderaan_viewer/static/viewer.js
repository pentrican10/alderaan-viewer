// ALDERAAN viewer client. The run comes from <body data-run-id>; the selected
// target is held in `current` and every request names both in its URL.

const RUN_ID = document.body.dataset.runId;
const PANELS = ["lightcurve", "folded", "single", "omc", "corner"];
const CORNER_PRESETS = {
    default: ["C0", "C1", "ROR", "IMPACT", "DUR14"],
    physical7: ["PERIOD", "EPOCH", "ROR", "IMPACT", "DUR14", "LD_U1", "LD_U2"],
    physical5: ["PERIOD", "EPOCH", "ROR", "IMPACT", "DUR14"],
    physical3: ["ROR", "IMPACT", "DUR14"],
    ld: ["LD_Q1", "LD_Q2", "LD_U1", "LD_U2"],
};

let current = null; // {koi, state, planets}
let singleStart = 0;

function api(path) {
    return `/api/${RUN_ID}/${current.koi}/${path}`;
}

async function errorText(r) {
    try {
        return (await r.json()).error;
    } catch {
        return `HTTP ${r.status}`;
    }
}

function showError(div, message) {
    Plotly.purge(div);
    div.textContent = message;
    div.className = "panel-error";
}

// Fetch a figure and draw it; returns the figure, or null on error
async function loadFigure(url, divId) {
    const div = document.getElementById(divId);
    div.className = "";
    div.textContent = "loading...";
    const koi = current.koi;
    const r = await fetch(url);
    if (current.koi !== koi) return null; // another target was selected meanwhile
    if (!r.ok) {
        showError(div, await errorText(r));
        return null;
    }
    const fig = await r.json();
    div.textContent = "";
    if (divId === "cornerPlot") squarePlotArea(div, fig.layout);
    await Plotly.react(div, fig.data, fig.layout, { responsive: true });
    return fig;
}

// Corner panels are square (axis constraints); a square plot area lets them
// fill it. The figure is 85% of the div width, and its height follows; reload
// the figure after a resize.
function squarePlotArea(div, layout) {
    const m = layout.margin;
    layout.width = Math.round(0.85 * div.clientWidth);
    layout.height = layout.width - (m.l + m.r) + (m.t + m.b);
}

// ---- target selection ----

async function selectTarget(koi) {
    document.querySelectorAll(".koi-row.selected").forEach(r => r.classList.remove("selected"));
    const row = document.getElementById(`row-${koi}`);
    row.classList.add("selected");

    current = { koi: koi, state: row.dataset.state, planets: [] };
    document.getElementById("koi_id_container").textContent = "";
    const h2 = document.createElement("h2");
    h2.textContent = `Selected koi_id: ${koi}`;
    document.getElementById("koi_id_container").appendChild(h2);

    const r = await fetch(api("status"));
    const banner = document.getElementById("target_banner");
    if (!r.ok) {
        banner.textContent = await errorText(r);
        banner.style.display = "block";
        return;
    }
    const status = await r.json();
    current.state = status.state;
    current.planets = status.planets;
    banner.textContent = status.state === "complete" ? "" : `${status.state}: ${status.message}`;
    banner.style.display = status.state === "complete" ? "none" : "block";

    for (const id of ["singlePlanetSelect", "cornerPlanetSelect"]) {
        const select = document.getElementById(id);
        select.textContent = "";
        for (const p of status.planets) {
            const opt = document.createElement("option");
            opt.value = p.n;
            opt.textContent = p.name;
            select.appendChild(opt);
        }
    }

    document.getElementById("checkbox_container").style.display = "block";
    document.getElementById("reviewContainer").style.display = "flex";
    document.getElementById("review_status").value =
        row.querySelector(".review-cell").textContent.trim();

    singleStart = 0;
    for (const panel of PANELS) updatePanel(panel);
    loadTables();
    loadComments();
}

function updatePanel(panel) {
    const shown = document.getElementById(`${panel}Checkbox`).checked;
    document.getElementById(`${panel}Panel`).style.display = shown ? "block" : "none";
    if (!shown || current === null) return;
    if (panel === "lightcurve") loadFigure(api("lightcurve"), "lightcurvePlot");
    if (panel === "folded") loadFigure(api("folded"), "foldedPlot");
    if (panel === "omc") loadFigure(api("omc"), "omcPlot");
    if (panel === "single") loadSingle();
    if (panel === "corner") loadCorner();
}

// ---- single transits ----

async function loadSingle() {
    const n = document.getElementById("singlePlanetSelect").value || 0;
    const fig = await loadFigure(api(`transits/${n}?start=${singleStart}`), "singlePlot");
    const page = document.getElementById("singlePage");
    page.textContent = "";
    if (fig !== null) {
        const m = fig.layout.meta;
        page.textContent = `transits ${m.start + 1}-${m.start + m.count} of ${m.n_transits}`;
        current.nTransits = m.n_transits;
    }
}

function moveSingle(step) {
    const next = singleStart + step;
    if (next < 0 || (current.nTransits !== undefined && next >= current.nTransits)) return;
    singleStart = next;
    loadSingle();
}

// ---- corner ----

function setCornerParams(params) {
    document.querySelectorAll("#cornerParams input").forEach(box => {
        box.checked = params.includes(box.value);
    });
}

function loadCorner() {
    const n = document.getElementById("cornerPlanetSelect").value || 0;
    const params = [...document.querySelectorAll("#cornerParams input:checked")].map(b => b.value);
    loadFigure(api(`corner/${n}?params=${params.join(",")}`), "cornerPlot");
}

// ---- tables ----

function fmt(v, digits) {
    return v === null || v === undefined ? "n/a" : Number(v).toFixed(digits);
}

function addRow(tbody, cells) {
    const tr = document.createElement("tr");
    for (const c of cells) {
        const td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
    }
    tbody.appendChild(tr);
}

async function loadTables() {
    const panel = document.getElementById("tablesPanel");
    const props = document.querySelector("#planetPropertiesTable tbody");
    const quality = document.querySelector("#transitQualityTable tbody");
    const ratios = document.querySelector("#periodRatiosTable tbody");
    const error = document.getElementById("tablesError");
    props.textContent = "";
    quality.textContent = "";
    ratios.textContent = "";
    error.textContent = "";
    panel.style.display = "block";

    const r = await fetch(api("tables"));
    if (!r.ok) {
        error.textContent = await errorText(r);
        return;
    }
    const data = await r.json();
    for (const p of data.properties) {
        if (p.not_modeled) {
            addRow(props, [p.name, "not modeled", "", "", ""]);
            continue;
        }
        addRow(props, [
            p.name,
            fmt(p.period, 6),
            fmt(p.ror, 4),
            fmt(p.impact, 3),
            fmt(p.dur14_hr, 3),
        ]);
        addRow(quality, [
            p.name,
            p.n_good,
            p.n_transits - p.n_good,
            p.n_transits,
            fmt(p.chi2_red, 2),
        ]);
    }
    if (data.period_ratios.length === 0) {
        addRow(ratios, ["single planet system", "", ""]);
    }
    for (const q of data.period_ratios) {
        addRow(ratios, [q.planets, q.periods, fmt(q.ratio, 4)]);
    }
}

// ---- review status and comments ----

async function setReviewStatus(status) {
    const r = await fetch(api("review"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: status }),
    });
    if (!r.ok) {
        alert(await errorText(r));
        return;
    }
    const row = document.getElementById(`row-${current.koi}`);
    row.className = `koi-row selected ${status.toLowerCase().replace(/ /g, "-")}`;
    row.querySelector(".review-cell").textContent = status;
}

async function showComments(r) {
    const container = document.getElementById("file_content_container");
    container.textContent = r.ok ? await r.text() : await errorText(r);
    document.getElementById("editTextarea").style.display = "none";
    document.getElementById("saveChangesButton").style.display = "none";
    document.getElementById("editButton").style.display = "block";
}

async function loadComments() {
    showComments(await fetch(api("comments")));
}

async function sendComments(method, body) {
    showComments(
        await fetch(api("comments"), {
            method: method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        })
    );
}

// ---- wiring ----

document.getElementById("runSelect").addEventListener("change", e => {
    window.location.href = `/home?run=${encodeURIComponent(e.target.value)}`;
});

document.getElementById("searchInput").addEventListener("input", e => {
    const q = e.target.value.trim().toUpperCase();
    document.querySelectorAll(".koi-row").forEach(row => {
        row.style.display = row.dataset.koiId.includes(q) ? "" : "none";
    });
});

document.querySelectorAll(".koi-id-cell a").forEach(a => {
    a.addEventListener("click", e => {
        e.preventDefault();
        selectTarget(a.closest("tr").dataset.koiId);
    });
});

for (const panel of PANELS) {
    document.getElementById(`${panel}Checkbox`).addEventListener("change", () => updatePanel(panel));
}

document.getElementById("singlePlanetSelect").addEventListener("change", () => {
    singleStart = 0;
    current.nTransits = undefined;
    loadSingle();
});
document.getElementById("singleBack").addEventListener("click", () => moveSingle(-9));
document.getElementById("singleForward").addEventListener("click", () => moveSingle(9));

document.getElementById("cornerPlanetSelect").addEventListener("change", loadCorner);
document.getElementById("cornerPreset").addEventListener("change", e => {
    setCornerParams(CORNER_PRESETS[e.target.value]);
    loadCorner();
});
document.querySelectorAll("#cornerParams input").forEach(box => box.addEventListener("change", loadCorner));
setCornerParams(CORNER_PRESETS.default);

document.getElementById("review_status").addEventListener("change", e => setReviewStatus(e.target.value));

document.getElementById("comment_form").addEventListener("submit", e => {
    e.preventDefault();
    const input = document.getElementById("commentInput");
    if (input.value.trim() === "") return;
    sendComments("POST", { comment: input.value });
    input.value = "";
});

document.getElementById("editButton").addEventListener("click", () => {
    const area = document.getElementById("editTextarea");
    area.value = document.getElementById("file_content_container").textContent;
    area.style.display = "block";
    document.getElementById("saveChangesButton").style.display = "block";
    document.getElementById("editButton").style.display = "none";
});

document.getElementById("saveChangesButton").addEventListener("click", () => {
    sendComments("PUT", { content: document.getElementById("editTextarea").value });
});
