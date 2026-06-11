/**
 * BerniniStudio - Editor JS v2.0
 *
 * Frontend for the BerniniStudio ComfyUI node.
 * Task-aware editor with guidance_mode display, system prompt preview,
 * Ollama prompt enhancement, and default negative prompt support.
 */
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

/* ── palette ────────────────────────────────────────────────────── */
const C = {
    bg:       "#18181b",
    panel:    "#27272a",
    deep:     "#111113",
    input:    "#0a0a0b",
    border:   "#3f3f46",
    borderIn: "#52525b",
    text:     "#e4e4e7",
    muted:    "#a1a1aa",
    dim:      "#71717a",
    accent:   "#f43f5e",
    accentDim:"#be123c",
    amber:    "#fbbf24",
    green:    "#4ade80",
    cyan:     "#22d3ee",
};

const HIDDEN_WIDGETS = [
    "task_type", "prompt", "negative_prompt",
    "ollama_url", "ollama_model", "use_default_neg",
    "api_format", "auto_enhance", "unload_ollama", "slot_images",
    "augment_strength", "augment_decay", "augment_seed",
];

/* ── per-task metadata ──────────────────────────────────────────── */
const TASK_META = {
    "v2v":   { label: "Video Edit",           wire: "source_video",                       guidance: "v2v_apg",
               desc: "General video editing: add, remove, or replace objects, restyle, change camera, colorize, inpaint. Prompt describes modifications + what to preserve." },
    "rv2v":  { label: "Ref + Video Edit",     wire: "source_video + image0",   guidance: "rv2v",
               desc: "Edit a video guided by reference image(s). The reference provides the appearance for replacements or additions.",
               refHint: "Reference images in your prompt as: image0, image1, image2 (not reference_image_0). e.g. 'Replace the man with the person from image0'." },
    "r2v":   { label: "Reference to Video",   wire: "image0",                  guidance: "r2v_apg",
               desc: "Generate a new video starring the subject(s) from your reference image(s). No source video needed.",
               refHint: "Reference subjects as: image0, image1, etc. e.g. 'The woman from image0 walks through a park'." },
    "t2v":   { label: "Text to Video",        wire: "none",                               guidance: "t2v_apg",
               desc: "Pure text-to-video generation. No source media. Describe the scene, subjects, motion, and camera work." },
    "t2i":   { label: "Text to Image",        wire: "none",                               guidance: "t2v_apg",
               desc: "Pure text-to-image generation. No source media. Describe composition, lighting, and subject matter." },
    "r2i":   { label: "Reference to Image",   wire: "image0",                  guidance: "r2v_apg",
               desc: "Generate a new still image featuring the subject(s) from your reference image(s).",
               refHint: "Reference subjects as: image0, image1, etc. e.g. 'The man from image0 sitting at a desk'." },
    "i2i":   { label: "Image to Image",       wire: "source_video (1 frame)",             guidance: "v2v",
               desc: "Standard image editing. Feed a single image as source_video. Add, remove, replace objects, restyle, relighting, inpaint, outpaint." },
    "i2v":   { label: "Image to Video",       wire: "image0",                  guidance: "r2v_apg",
               desc: "Animate a static reference image into a video.",
               refHint: "Reference the image as: image0. e.g. 'The scene from image0 comes to life as the camera slowly pushes in'." },
    "mv2v":  { label: "Motion Edit",          wire: "source_video",                       guidance: "v2v_apg",
               desc: "Change the motion, pose, or action of subjects in a video while preserving their identity and the scene. e.g. 'the person begins dancing'." },
    "vi2v":  { label: "Content Propagation",  wire: "source_video + image0",   guidance: "v2v_apg",
               desc: "Propagate an edit from the first frame to the full video, or composite reference content into the source video.",
               refHint: "Reference the content as: image0. e.g. 'Integrate the object from image0 into the video'." },
    "ads2v": { label: "Ads Insertion",        wire: "source_video + image0",   guidance: "v2v_apg",
               desc: "Insert a logo, ad, or branded element into a video scene. Bernini matches perspective, lighting, and occlusion.",
               refHint: "The ad/logo is image0. e.g. 'Add the image0 logo on the billboard on the left'." },
    "vrc2v": { label: "Video Retarget",       wire: "source_video + image0",   guidance: "rv2v",
               desc: "Retarget or adjust a subject's position, action, or framing using reference guidance.",
               refHint: "Reference the guide as: image0. e.g. 'Match the pose of the person in image0'." },
};

const SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": "You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
    "mv2v": "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, colors, textures, and the subject's pose or action.",
};

/* ── helpers ────────────────────────────────────────────────────── */

function hideWidget(w) {
    if (!w) return;
    // NOTE: do NOT override w.type -- changing it breaks ComfyUI's widget
    // serialization and causes values (e.g. task_type) to reset on tab
    // switch / reload (GitHub issue #1). Visual hiding via computeSize/draw
    // is sufficient and keeps serialization intact.
    w.__bernini_hidden = true;
    w.computeSize = () => [0, -4];
    w.draw = () => {};
    const elems = [w.inputEl, w.element].filter(Boolean);
    for (const el of elems) {
        el.style.display = "none";
        el.hidden = true;
        let p = el.parentElement;
        let hops = 0;
        while (p && hops < 6) {
            if (p.classList && p.classList.contains("dom-widget")) {
                p.style.display = "none";
                p.style.height = "0px";
                p.style.minHeight = "0px";
                p.style.padding = "0";
                p.style.margin = "0";
                break;
            }
            p = p.parentElement;
            hops++;
        }
    }
}

function findWidget(node, name) {
    return (node.widgets || []).find(w => w.name === name);
}

function swallowKeys(el) {
    el.addEventListener("keydown", e => e.stopPropagation());
    el.addEventListener("keyup", e => e.stopPropagation());
    el.addEventListener("keypress", e => e.stopPropagation());
}

function el(tag, styles, text) {
    const e = document.createElement(tag);
    if (styles) Object.assign(e.style, styles);
    if (text !== undefined) e.textContent = text;
    return e;
}

/* ── Editor class ───────────────────────────────────────────────── */

class BerniniEditor {
    constructor(node, mount) {
        this.node = node;
        this.mount = mount;
        node._berniniEditor = this;  // stored for onConfigure re-sync
        this.taskWidget = findWidget(node, "task_type");
        this.promptWidget = findWidget(node, "prompt");
        this.negWidget = findWidget(node, "negative_prompt");
        this.urlWidget = findWidget(node, "ollama_url");
        this.modelWidget = findWidget(node, "ollama_model");
        this.apiFormatWidget = findWidget(node, "api_format");
        this.autoEnhanceWidget = findWidget(node, "auto_enhance");
        this.unloadOllamaWidget = findWidget(node, "unload_ollama");
        this.slotImagesWidget = findWidget(node, "slot_images");
        this.augStrengthWidget = findWidget(node, "augment_strength");
        this.augDecayWidget = findWidget(node, "augment_decay");
        this.augSeedWidget = findWidget(node, "augment_seed");
        this.useDefNegWidget = findWidget(node, "use_default_neg");
        this.ollamaOpen = false;
        this._build();
    }

    _si(el) { // style input
        Object.assign(el.style, {
            width: "100%", minWidth: "0", boxSizing: "border-box",
            background: C.input, color: C.text,
            border: "1px solid " + C.borderIn, borderRadius: "4px",
            padding: "4px 6px", fontFamily: "inherit", fontSize: "11px", outline: "none",
        });
        el.onfocus = () => el.style.borderColor = C.accent;
        el.onblur  = () => el.style.borderColor = C.borderIn;
    }

    _sta(el) { // style textarea
        this._si(el);
        el.style.resize = "vertical";
    }

    _build() {
        for (const name of HIDDEN_WIDGETS) hideWidget(findWidget(this.node, name));

        this.root = el("div", {
            display: "flex", flexDirection: "column", gap: "8px",
            background: C.bg, color: C.text, padding: "10px",
            borderRadius: "6px", fontFamily: "ui-sans-serif, system-ui, sans-serif",
            fontSize: "12px", width: "100%", boxSizing: "border-box",
            boxShadow: "inset 0 2px 5px rgba(0,0,0,0.5)",
        });
        this.mount.appendChild(this.root);

        /* ── header ──────────────────────────────────────────────── */
        const header = el("div", {
            display: "flex", justifyContent: "space-between", alignItems: "center",
            paddingBottom: "6px", borderBottom: "1px solid " + C.border,
        });
        header.appendChild(el("div", {
            fontWeight: "700", fontSize: "14px", color: C.accent, letterSpacing: "0.5px",
        }, "Bernini Studio"));
        const badges = el("div", { display: "flex", gap: "4px", alignItems: "center" });
        this.guidanceBadge = el("span", {
            fontSize: "9px", color: C.cyan, background: "rgba(34,211,238,0.1)",
            padding: "2px 6px", borderRadius: "3px", border: "1px solid rgba(34,211,238,0.25)",
            fontFamily: "monospace",
        }, "");
        badges.appendChild(this.guidanceBadge);
        badges.appendChild(el("span", {
            fontSize: "9px", color: C.muted, background: C.deep,
            padding: "2px 6px", borderRadius: "3px", border: "1px solid " + C.borderIn,
        }, "v2.2"));
        header.appendChild(badges);
        this.root.appendChild(header);

        /* ── task selector ───────────────────────────────────────── */
        const taskBox = el("div", {
            background: C.panel, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "8px",
            display: "flex", flexDirection: "column", gap: "5px",
        });
        taskBox.appendChild(el("div", {
            fontWeight: "600", fontSize: "10px", color: C.muted,
            textTransform: "uppercase", letterSpacing: "0.5px",
        }, "Task Mode"));

        this.taskSelect = document.createElement("select");
        this._si(this.taskSelect);
        if (this.taskWidget && this.taskWidget.options && this.taskWidget.options.values) {
            for (const v of this.taskWidget.options.values) {
                const opt = document.createElement("option");
                opt.value = v;
                const meta = TASK_META[v] || {};
                opt.textContent = v + " - " + (meta.label || v);
                if (this.taskWidget.value === v) opt.selected = true;
                this.taskSelect.appendChild(opt);
            }
        }
        taskBox.appendChild(this.taskSelect);

        // hint row: wire + guidance
        this.hintBox = el("div", {
            fontSize: "10px", color: C.amber,
            background: "rgba(251,191,36,0.06)", padding: "5px 8px",
            borderRadius: "3px", border: "1px solid rgba(251,191,36,0.15)",
            lineHeight: "1.5", display: "flex", flexDirection: "column", gap: "2px",
        });
        taskBox.appendChild(this.hintBox);

        this.taskSelect.onchange = () => {
            if (this.taskWidget) this.taskWidget.value = this.taskSelect.value;
            this._updateTaskDisplay();
        };

        this.root.appendChild(taskBox);

        /* ── system prompt preview ───────────────────────────────── */
        const sysBox = el("div", {
            background: C.deep, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "6px 8px",
            display: "flex", flexDirection: "column", gap: "3px",
        });
        const sysHeader = el("div", {
            display: "flex", justifyContent: "space-between", alignItems: "center",
            cursor: "pointer", userSelect: "none",
        });
        const sysLabel = el("div", {
            fontWeight: "600", fontSize: "10px", color: C.dim,
            textTransform: "uppercase", letterSpacing: "0.5px",
        }, "System Prompt (auto-prepended)");
        this.sysArrow = el("span", { fontSize: "9px", color: C.dim, transition: "transform 0.2s", marginRight: "4px" }, "\u25B6");
        const sysLeft = el("div", { display: "flex", alignItems: "center", gap: "4px" });
        sysLeft.appendChild(this.sysArrow);
        sysLeft.appendChild(sysLabel);
        sysHeader.appendChild(sysLeft);

        const copyBtn = el("button", {
            background: "transparent", color: C.dim, border: "1px solid " + C.borderIn,
            borderRadius: "3px", padding: "1px 6px", fontSize: "9px", cursor: "pointer",
        }, "Copy");
        copyBtn.onclick = (e) => {
            e.stopPropagation();
            const sp = SYSTEM_PROMPTS[this.taskSelect.value] || SYSTEM_PROMPTS["default"];
            navigator.clipboard.writeText(sp).then(() => {
                copyBtn.textContent = "Copied";
                setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
            });
        };
        sysHeader.appendChild(copyBtn);
        sysBox.appendChild(sysHeader);

        this.sysPromptEl = el("div", {
            fontSize: "10px", color: C.muted, fontStyle: "italic",
            lineHeight: "1.4", wordBreak: "break-word",
        });
        sysBox.appendChild(this.sysPromptEl);

        // Expandable enhancement template editor
        this.templateBox = el("div", {
            display: "none", flexDirection: "column", gap: "4px",
            marginTop: "4px", borderTop: "1px solid " + C.borderIn, paddingTop: "6px",
        });
        this.templateBox.appendChild(el("div", {
            fontSize: "9px", color: C.dim,
        }, "LLM Enhancement Template (editable -- changes apply to next Enhance click):"));
        this.templateArea = document.createElement("textarea");
        this.templateArea.rows = 8;
        this._sta(this.templateArea);
        this.templateArea.style.fontSize = "9px";
        this.templateArea.style.lineHeight = "1.4";
        swallowKeys(this.templateArea);

        const templateBtnRow = el("div", { display: "flex", gap: "4px", justifyContent: "flex-end" });
        const resetBtn = el("button", {
            background: "transparent", color: C.dim, border: "1px solid " + C.borderIn,
            borderRadius: "3px", padding: "1px 8px", fontSize: "9px", cursor: "pointer",
        }, "Reset to Default");
        resetBtn.onclick = () => {
            this.templateArea.value = this._currentDefaultTemplate || "";
        };
        templateBtnRow.appendChild(resetBtn);
        this.templateBox.appendChild(this.templateArea);
        this.templateBox.appendChild(templateBtnRow);
        sysBox.appendChild(this.templateBox);

        this._templateOpen = false;
        sysHeader.onclick = () => {
            this._templateOpen = !this._templateOpen;
            this.templateBox.style.display = this._templateOpen ? "flex" : "none";
            this.sysArrow.style.transform = this._templateOpen ? "rotate(90deg)" : "rotate(0deg)";
            if (this._templateOpen && !this.templateArea.value) {
                this._fetchTemplate();
            }
        };

        this.root.appendChild(sysBox);

        /* ── prompt areas ────────────────────────────────────────── */
        const promptBox = el("div", {
            background: C.panel, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "8px",
            display: "flex", flexDirection: "column", gap: "5px",
        });
        promptBox.appendChild(el("div", {
            fontWeight: "600", fontSize: "10px", color: C.muted,
            textTransform: "uppercase", letterSpacing: "0.5px",
        }, "Prompt"));

        this.promptArea = document.createElement("textarea");
        this.promptArea.rows = 4;
        this._sta(this.promptArea);
        swallowKeys(this.promptArea);
        this.promptArea.value = this.promptWidget ? this.promptWidget.value : "";
        this.promptArea.placeholder = "Describe your edit or generation...";
        this.promptArea.oninput = () => {
            if (this.promptWidget) this.promptWidget.value = this.promptArea.value;
        };
        promptBox.appendChild(this.promptArea);

        // negative prompt
        const negRow = el("div", { display: "flex", justifyContent: "space-between", alignItems: "center" });
        negRow.appendChild(el("div", {
            fontSize: "10px", color: C.muted,
        }, "Negative Prompt"));

        // default neg toggle
        const defNegLabel = el("label", {
            display: "flex", alignItems: "center", gap: "4px",
            fontSize: "9px", color: C.dim, cursor: "pointer",
        });
        this.defNegCheck = document.createElement("input");
        this.defNegCheck.type = "checkbox";
        this.defNegCheck.checked = this.useDefNegWidget ? this.useDefNegWidget.value : true;
        this.defNegCheck.style.cssText = "width:12px;height:12px;cursor:pointer;accent-color:" + C.accent;
        this.defNegCheck.onchange = () => {
            if (this.useDefNegWidget) this.useDefNegWidget.value = this.defNegCheck.checked;
            this._updateNegPlaceholder();
        };
        defNegLabel.appendChild(this.defNegCheck);
        defNegLabel.appendChild(document.createTextNode("Use Bernini default if empty"));
        negRow.appendChild(defNegLabel);
        promptBox.appendChild(negRow);

        this.negArea = document.createElement("textarea");
        this.negArea.rows = 2;
        this._sta(this.negArea);
        swallowKeys(this.negArea);
        this.negArea.value = this.negWidget ? this.negWidget.value : "";
        this.negArea.oninput = () => {
            if (this.negWidget) this.negWidget.value = this.negArea.value;
        };
        promptBox.appendChild(this.negArea);
        this.root.appendChild(promptBox);

        /* ── reference image slots (collapsible, drag & drop) ────── */
        const slotHeader = el("div", {
            display: "flex", justifyContent: "space-between", alignItems: "center",
            background: C.deep, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "6px 8px", cursor: "pointer",
            userSelect: "none",
        });
        const slotTitleRow = el("div", { display: "flex", alignItems: "center", gap: "6px" });
        slotTitleRow.appendChild(el("div", {
            fontWeight: "600", fontSize: "10px", color: C.dim,
            textTransform: "uppercase", letterSpacing: "0.5px",
        }, "Reference Images (drop-in)"));
        this.slotCountBadge = el("span", {
            fontSize: "9px", color: C.cyan, background: "rgba(34,211,238,0.1)",
            padding: "1px 6px", borderRadius: "3px", border: "1px solid rgba(34,211,238,0.25)",
            fontFamily: "monospace", display: "none",
        }, "");
        slotTitleRow.appendChild(this.slotCountBadge);
        slotHeader.appendChild(slotTitleRow);
        this.slotArrow = el("span", { fontSize: "10px", color: C.dim, transition: "transform 0.2s" }, "\u25B6");
        slotHeader.appendChild(this.slotArrow);
        this.root.appendChild(slotHeader);

        this.slotBody = el("div", {
            background: C.deep, border: "1px solid " + C.borderIn,
            borderTop: "none", borderRadius: "0 0 4px 4px",
            padding: "8px", display: "none",
            flexDirection: "column", gap: "6px", marginTop: "-5px",
        });
        this.slotBody.appendChild(el("div", { fontSize: "9px", color: C.dim },
            "click / drop to load -- drag between slots to reorder -- wired jacks override"));

        this.slotGrid = el("div", {
            display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "6px",
        });
        this.slotBody.appendChild(this.slotGrid);
        this.root.appendChild(this.slotBody);

        this._slotOpen = false;
        slotHeader.onclick = () => {
            this._slotOpen = !this._slotOpen;
            this.slotBody.style.display = this._slotOpen ? "flex" : "none";
            this.slotArrow.style.transform = this._slotOpen ? "rotate(90deg)" : "rotate(0deg)";
            requestAnimationFrame(() => {
                if (this.node.computeSize && this.node.size) this.node.size[1] = this.node.computeSize()[1];
                if (this.node.graph) this.node.graph.setDirtyCanvas(true, true);
            });
        };

        // hidden file input shared by all slots
        this.slotFileInput = document.createElement("input");
        this.slotFileInput.type = "file";
        this.slotFileInput.accept = "image/*";
        this.slotFileInput.style.display = "none";
        this.root.appendChild(this.slotFileInput);

        this._slotFiles = this._readSlotWidget();
        this._renderSlots();

        /* ── advanced (experimental) ─────────────────────────────── */
        const advHeader = el("div", {
            display: "flex", justifyContent: "space-between", alignItems: "center",
            background: C.deep, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "6px 8px", cursor: "pointer",
            userSelect: "none",
        });
        advHeader.appendChild(el("div", {
            fontWeight: "600", fontSize: "10px", color: C.dim,
            textTransform: "uppercase", letterSpacing: "0.5px",
        }, "Advanced (experimental)"));
        this.advArrow = el("span", { fontSize: "10px", color: C.dim, transition: "transform 0.2s" }, "\u25B6");
        advHeader.appendChild(this.advArrow);
        this.root.appendChild(advHeader);

        this.advBody = el("div", {
            background: C.deep, border: "1px solid " + C.borderIn,
            borderTop: "none", borderRadius: "0 0 4px 4px",
            padding: "8px", display: "none",
            flexDirection: "column", gap: "8px", marginTop: "-5px",
        });
        this._advOpen = false;
        advHeader.onclick = () => {
            this._advOpen = !this._advOpen;
            this.advBody.style.display = this._advOpen ? "flex" : "none";
            this.advArrow.style.transform = this._advOpen ? "rotate(90deg)" : "rotate(0deg)";
            requestAnimationFrame(() => {
                if (this.node.computeSize && this.node.size) this.node.size[1] = this.node.computeSize()[1];
                if (this.node.graph) this.node.graph.setDirtyCanvas(true, true);
            });
        };

        this.advBody.appendChild(el("div", { fontSize: "9px", color: C.dim, lineHeight: "1.4" },
            "Augment empty latent: adds seeded noise to the init latent. Bernini was trained on a clean " +
            "empty latent, so 0 = standard behavior. Low values (0.05-0.2) are worth trying for motion/detail variation."));

        const mkSlider = (label, widget, min, max, step) => {
            const row = el("div", { display: "flex", alignItems: "center", gap: "8px" });
            row.appendChild(el("span", { fontSize: "10px", color: C.muted, minWidth: "70px" }, label));
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = min; slider.max = max; slider.step = step;
            // Coerce undefined/NaN to 0 -- an unset range input defaults to its
            // midpoint (0.5), which would silently enable experimental features
            // on workflows saved before these widgets existed.
            const wv = widget ? parseFloat(widget.value) : 0;
            slider.value = isNaN(wv) ? 0 : wv;
            if (widget && isNaN(parseFloat(widget.value))) widget.value = 0;
            slider.style.cssText = "flex:1;accent-color:" + C.accent + ";cursor:pointer;height:14px;";
            const valEl = el("span", {
                fontSize: "10px", color: C.cyan, fontFamily: "monospace",
                minWidth: "38px", textAlign: "right",
            }, String(slider.value));
            slider.oninput = () => {
                if (widget) widget.value = parseFloat(slider.value);
                valEl.textContent = slider.value;
            };
            row.appendChild(slider);
            row.appendChild(valEl);
            row._slider = slider;
            row._valEl = valEl;
            this.advBody.appendChild(row);
            return row;
        };

        this.augStrengthRow = mkSlider("Strength", this.augStrengthWidget, 0, 1, 0.01);
        this.augDecayRow = mkSlider("Decay", this.augDecayWidget, 0, 1, 0.01);

        const seedRow = el("div", { display: "flex", alignItems: "center", gap: "8px" });
        seedRow.appendChild(el("span", { fontSize: "10px", color: C.muted, minWidth: "70px" }, "Seed"));
        this.augSeedInput = document.createElement("input");
        this.augSeedInput.type = "number";
        this.augSeedInput.min = 0;
        this._si(this.augSeedInput);
        this.augSeedInput.style.flex = "1";
        this.augSeedInput.value = this.augSeedWidget ? this.augSeedWidget.value : 0;
        this.augSeedInput.oninput = () => {
            if (this.augSeedWidget) this.augSeedWidget.value = parseInt(this.augSeedInput.value || "0", 10);
        };
        swallowKeys(this.augSeedInput);
        seedRow.appendChild(this.augSeedInput);
        this.advBody.appendChild(seedRow);

        this.advBody.appendChild(el("div", {
            fontSize: "9px", color: C.dim, lineHeight: "1.4",
            borderTop: "1px solid " + C.borderIn, paddingTop: "6px",
        }, "NAG at CFG 1: native ComfyUI skips the negative prompt entirely at cfg 1.0. " +
           "For attention-level negative guidance, wire this node's positive/negative into " +
           "ComfyUI-NAG's NAGCFGGuider and sample with SamplerCustomAdvanced -- it composes " +
           "directly with BerniniStudio's outputs."));

        this.root.appendChild(this.advBody);

        /* ── Ollama enhancer (collapsible) ───────────────────────── */
        const ollamaHeader = el("div", {
            display: "flex", justifyContent: "space-between", alignItems: "center",
            background: C.deep, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "6px 8px", cursor: "pointer",
            userSelect: "none",
        });
        const ollamaTitle = el("div", {
            fontWeight: "600", fontSize: "10px", color: C.dim,
            textTransform: "uppercase", letterSpacing: "0.5px",
        }, "LLM Prompt Enhancer");
        this.ollamaArrow = el("span", { fontSize: "10px", color: C.dim, transition: "transform 0.2s" }, "\u25B6");
        ollamaHeader.appendChild(ollamaTitle);
        ollamaHeader.appendChild(this.ollamaArrow);
        ollamaHeader.onclick = () => this._toggleOllama();
        this.root.appendChild(ollamaHeader);

        this.ollamaBody = el("div", {
            background: C.deep, border: "1px solid " + C.borderIn,
            borderTop: "none", borderRadius: "0 0 4px 4px",
            padding: "8px", display: "none",
            flexDirection: "column", gap: "6px", marginTop: "-5px",
        });

        this.ollamaBody.appendChild(el("div", {
            fontSize: "9px", color: C.dim, lineHeight: "1.4",
        }, "Rewrites your short instruction using Bernini's official per-task prompt templates. With a vision model, reference images are sent to the LLM for accurate description."));

        // API format toggle
        const formatRow = el("div", { display: "flex", gap: "6px", alignItems: "center" });
        formatRow.appendChild(el("div", { fontSize: "10px", color: C.muted, whiteSpace: "nowrap" }, "API:"));
        this.apiFormatSelect = document.createElement("select");
        this._si(this.apiFormatSelect);
        this.apiFormatSelect.style.flex = "0 0 auto";
        this.apiFormatSelect.style.width = "auto";
        for (const [val, label] of [["Ollama", "Ollama (/api/chat)"], ["OpenAI / vLLM", "OpenAI / vLLM (/v1/chat)"]]) {
            const o = document.createElement("option");
            o.value = val; o.textContent = label;
            if (this.apiFormatWidget && this.apiFormatWidget.value === val) o.selected = true;
            this.apiFormatSelect.appendChild(o);
        }
        this.apiFormatSelect.onchange = () => {
            if (this.apiFormatWidget) this.apiFormatWidget.value = this.apiFormatSelect.value;
            this._fetchModels();
        };
        formatRow.appendChild(this.apiFormatSelect);

        // Vision indicator
        this.visionBadge = el("span", {
            fontSize: "9px", color: C.green, background: "rgba(74,222,128,0.1)",
            padding: "1px 6px", borderRadius: "3px", border: "1px solid rgba(74,222,128,0.2)",
            marginLeft: "auto", display: "none",
        }, "");
        formatRow.appendChild(this.visionBadge);
        this.ollamaBody.appendChild(formatRow);

        const urlRow = el("div", { display: "flex", gap: "6px" });
        this.urlInput = document.createElement("input");
        this.urlInput.type = "text";
        this.urlInput.placeholder = "http://127.0.0.1:11434";
        this._si(this.urlInput);
        this.urlInput.style.flex = "1 1 auto";
        this.urlInput.style.minWidth = "0";
        this.urlInput.value = this.urlWidget ? this.urlWidget.value : "http://127.0.0.1:11434";
        this.urlInput.oninput = () => { if (this.urlWidget) this.urlWidget.value = this.urlInput.value; };
        swallowKeys(this.urlInput);
        urlRow.appendChild(this.urlInput);

        const refreshBtn = el("button", {
            background: C.panel, color: C.text, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "2px 8px", fontSize: "10px",
            cursor: "pointer", whiteSpace: "nowrap",
        }, "Refresh");
        refreshBtn.onclick = () => this._fetchModels();
        urlRow.appendChild(refreshBtn);
        this.ollamaBody.appendChild(urlRow);

        this.modelSelect = document.createElement("select");
        this._si(this.modelSelect);
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "(Select a model)";
        this.modelSelect.appendChild(blank);
        if (this.modelWidget && this.modelWidget.value) {
            const o = document.createElement("option");
            o.value = this.modelWidget.value;
            o.textContent = this.modelWidget.value;
            o.selected = true;
            this.modelSelect.appendChild(o);
        }
        this.modelSelect.onchange = () => {
            if (this.modelWidget) this.modelWidget.value = this.modelSelect.value;
        };
        this.ollamaBody.appendChild(this.modelSelect);

        this.enhanceBtn = el("button", {
            background: C.accent, color: "white", border: "none",
            borderRadius: "4px", padding: "6px", fontWeight: "600",
            fontSize: "11px", cursor: "pointer", transition: "background 0.3s, opacity 0.2s",
        }, "Enhance Prompt");
        this.enhanceBtn.onmouseover = () => { if (!this.enhanceBtn.disabled) this.enhanceBtn.style.opacity = "0.85"; };
        this.enhanceBtn.onmouseout  = () => { if (!this.enhanceBtn.disabled) this.enhanceBtn.style.opacity = "1"; };
        this.enhanceBtn.onclick = () => this._enhancePrompt();
        this.ollamaBody.appendChild(this.enhanceBtn);

        // Auto-enhance toggle
        const autoRow = el("div", {
            display: "flex", alignItems: "center", gap: "6px", marginTop: "2px",
        });
        this.autoEnhanceCheck = document.createElement("input");
        this.autoEnhanceCheck.type = "checkbox";
        this.autoEnhanceCheck.checked = this.autoEnhanceWidget ? this.autoEnhanceWidget.value : false;
        this.autoEnhanceCheck.style.cssText = "width:12px;height:12px;cursor:pointer;accent-color:" + C.accent;
        this.autoEnhanceCheck.onchange = () => {
            if (this.autoEnhanceWidget) this.autoEnhanceWidget.value = this.autoEnhanceCheck.checked;
        };
        autoRow.appendChild(this.autoEnhanceCheck);
        autoRow.appendChild(el("span", { fontSize: "10px", color: C.muted, cursor: "pointer" },
            "Auto-enhance on queue (check ComfyUI console for enhanced text)"));
        autoRow.onclick = (e) => {
            if (e.target !== this.autoEnhanceCheck) {
                this.autoEnhanceCheck.checked = !this.autoEnhanceCheck.checked;
                this.autoEnhanceCheck.dispatchEvent(new Event("change"));
            }
        };
        this.ollamaBody.appendChild(autoRow);

        // Unload Ollama toggle + instant unload button
        const unloadRow = el("div", {
            display: "flex", alignItems: "center", gap: "6px",
        });
        this.unloadCheck = document.createElement("input");
        this.unloadCheck.type = "checkbox";
        this.unloadCheck.checked = this.unloadOllamaWidget ? this.unloadOllamaWidget.value : false;
        this.unloadCheck.style.cssText = "width:12px;height:12px;cursor:pointer;accent-color:" + C.accent;
        this.unloadCheck.onchange = () => {
            if (this.unloadOllamaWidget) this.unloadOllamaWidget.value = this.unloadCheck.checked;
        };
        unloadRow.appendChild(this.unloadCheck);
        const unloadLabel = el("span", { fontSize: "10px", color: C.muted, cursor: "pointer", flex: "1" },
            "Auto-unload after enhance (Ollama only)");
        unloadLabel.onclick = () => {
            this.unloadCheck.checked = !this.unloadCheck.checked;
            this.unloadCheck.dispatchEvent(new Event("change"));
        };
        unloadRow.appendChild(unloadLabel);

        this.unloadBtn = el("button", {
            background: C.panel, color: C.text, border: "1px solid " + C.borderIn,
            borderRadius: "4px", padding: "2px 10px", fontSize: "10px",
            cursor: "pointer", whiteSpace: "nowrap",
        }, "Unload Now");
        this.unloadBtn.onclick = () => this._unloadOllama();
        unloadRow.appendChild(this.unloadBtn);
        this.ollamaBody.appendChild(unloadRow);

        this.statusEl = el("div", {
            fontSize: "10px", color: C.dim, minHeight: "14px", textAlign: "center",
        });
        this.ollamaBody.appendChild(this.statusEl);
        this.root.appendChild(this.ollamaBody);

        // Inject CSS pulse animation (once, globally)
        if (!document.getElementById("bernini-pulse-css")) {
            const style = document.createElement("style");
            style.id = "bernini-pulse-css";
            style.textContent = `
                @keyframes bernini-pulse {
                    0%, 100% { background: #d97706; transform: scale(1); }
                    50% { background: #92400e; transform: scale(1.01); }
                }
            `;
            document.head.appendChild(style);
        }

        /* ── finalize ────────────────────────────────────────────── */
        this._updateTaskDisplay();
        this._updateNegPlaceholder();
        this._fetchModels(true);
        this._installVisibilityWatchdog();

        // Listen for server-side auto-enhance results pushed from execute()
        const nodeId = String(this.node.id);
        api.addEventListener("bernini_enhanced", (event) => {
            const d = event.detail;
            if (d && String(d.node) === nodeId && d.text) {
                this.promptArea.value = d.text;
                if (this.promptWidget) this.promptWidget.value = d.text;
                this.statusEl.textContent = "Auto-enhanced on queue (" + d.text.length + " chars)";
            }
        });
    }

    _updateTaskDisplay() {
        const task = this.taskSelect.value;
        const meta = TASK_META[task] || {};
        const sysp = SYSTEM_PROMPTS[task] || SYSTEM_PROMPTS["default"];

        // guidance badge
        this.guidanceBadge.textContent = meta.guidance || "?";

        // hint box: description + ref format hint + wire + guidance
        this.hintBox.innerHTML = "";
        if (meta.desc) {
            const descLine = el("div", {
                color: C.text, fontSize: "10px", lineHeight: "1.4",
                paddingBottom: "3px", marginBottom: "3px",
                borderBottom: "1px solid rgba(251,191,36,0.12)",
            }, meta.desc);
            this.hintBox.appendChild(descLine);
        }
        if (meta.refHint) {
            const refLine = el("div", {
                color: C.cyan, fontSize: "10px", lineHeight: "1.4",
                paddingBottom: "3px", marginBottom: "3px",
                borderBottom: "1px solid rgba(251,191,36,0.12)",
            }, meta.refHint);
            this.hintBox.appendChild(refLine);
        }
        const wireText = meta.wire === "none" ? "No media inputs needed" : "Wire: " + meta.wire;
        const infoRow = el("div", {
            display: "flex", justifyContent: "space-between", alignItems: "center",
            flexWrap: "wrap", gap: "4px",
        });
        infoRow.appendChild(el("span", { color: C.amber, fontSize: "10px" }, wireText));
        infoRow.appendChild(el("span", { color: C.dim, fontSize: "9px", fontFamily: "monospace" },
            "guidance: " + (meta.guidance || "?")));
        this.hintBox.appendChild(infoRow);

        // system prompt preview
        this.sysPromptEl.textContent = sysp;

        // dynamic prompt placeholder based on task type
        if (meta.refHint) {
            this.promptArea.placeholder = "e.g. 'Replace the man with the person from image0' (use image0, image1... not reference_image_0)";
        } else if (meta.wire === "none") {
            this.promptArea.placeholder = "Describe the scene, subjects, and action...";
        } else {
            this.promptArea.placeholder = "Describe your edit (what to change + what to preserve)...";
        }

        // Refresh template if the template editor is open
        if (this._templateOpen) {
            this._fetchTemplate();
        }
    }

    async _fetchTemplate() {
        try {
            const resp = await api.fetchApi("/bernini_studio/get_template", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ task_type: this.taskSelect.value }),
            });
            const data = await resp.json();
            if (data.template) {
                this._currentDefaultTemplate = data.template;
                this.templateArea.value = data.template;
            }
        } catch (e) {
            console.warn("[BerniniStudio] Failed to fetch template:", e);
        }
    }

    _updateNegPlaceholder() {
        if (this.defNegCheck.checked && !this.negArea.value.trim()) {
            this.negArea.placeholder = "(Bernini default Chinese neg will be used)";
        } else {
            this.negArea.placeholder = "";
        }
    }

    _toggleOllama() {
        this.ollamaOpen = !this.ollamaOpen;
        this.ollamaBody.style.display = this.ollamaOpen ? "flex" : "none";
        this.ollamaArrow.style.transform = this.ollamaOpen ? "rotate(90deg)" : "rotate(0deg)";
    }

    async _fetchModels(silent) {
        if (!silent) this.statusEl.textContent = "Fetching models...";
        const fmt = this.apiFormatSelect ? this.apiFormatSelect.value : "ollama";
        try {
            const resp = await api.fetchApi("/bernini_studio/models", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ollama_url: this.urlInput.value,
                    api_format: fmt,
                }),
            });
            const data = await resp.json();
            if (!resp.ok) {
                if (!silent) this.statusEl.textContent = (fmt === "openai" ? "vLLM" : "Ollama") + ": " + (data.error || resp.status);
                return;
            }
            const models = data.models || [];
            this.modelSelect.innerHTML = "";
            const b = document.createElement("option");
            b.value = "";
            b.textContent = "(Select a model)";
            this.modelSelect.appendChild(b);
            for (const m of models) {
                const o = document.createElement("option");
                o.value = m;
                o.textContent = m;
                this.modelSelect.appendChild(o);
            }
            const prior = this.modelWidget ? this.modelWidget.value : "";
            if (prior && models.includes(prior)) this.modelSelect.value = prior;
            if (!silent) this.statusEl.textContent = models.length + " model(s) found.";
        } catch (e) {
            if (!silent) this.statusEl.textContent = "Could not reach server.";
        }
    }

    async _enhancePrompt() {
        const model = this.modelSelect.value;
        if (!model) { this.statusEl.textContent = "Select a model first."; return; }
        if (!this.promptArea.value.trim()) { this.statusEl.textContent = "Enter an instruction to enhance."; return; }

        this._setEnhanceBusy(true, "Collecting images...");
        const fmt = this.apiFormatSelect ? this.apiFormatSelect.value : "Ollama";

        // Collect source video frames + reference images
        let images = [];
        let sourceFrameCount = 0;
        try {
            const sourceFrames = await this._collectSourceVideoFrames();
            sourceFrameCount = sourceFrames.length;
            images = images.concat(sourceFrames);
        } catch (e) {
            console.warn("[BerniniStudio] Source frame extraction failed:", e);
        }
        try {
            const refImages = await this._collectReferenceImages();
            images = images.concat(refImages);
        } catch (e) {
            console.warn("[BerniniStudio] Reference image collection failed:", e);
        }

        const refCount = images.length - sourceFrameCount;
        if (images.length > 0) {
            let parts = [];
            if (sourceFrameCount > 0) parts.push(sourceFrameCount + " video frame(s)");
            if (refCount > 0) parts.push(refCount + " ref img(s)");
            this.visionBadge.textContent = parts.join(" + ");
            this.visionBadge.style.display = "inline";
            this._setEnhanceBusy(true, "Sending " + parts.join(" + ") + " to " + model + "...");
        } else {
            this.visionBadge.style.display = "none";
            this._setEnhanceBusy(true, "Asking " + model + " (text-only)...");
        }

        // Check for custom template override
        const customTemplate = (this.templateArea && this.templateArea.value.trim() !== this._currentDefaultTemplate)
            ? this.templateArea.value.trim() : "";

        try {
            const resp = await api.fetchApi("/bernini_studio/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ollama_url: this.urlInput.value,
                    model: model,
                    prompt: this.promptArea.value,
                    task_type: this.taskSelect.value,
                    image_num: Math.max(1, refCount),
                    images: images,
                    api_format: fmt,
                    unload_ollama: this.unloadCheck ? this.unloadCheck.checked : false,
                    custom_template: customTemplate,
                }),
            });
            const data = await resp.json();
            if (data.response) {
                this.promptArea.value = data.response;
                if (this.promptWidget) this.promptWidget.value = data.response;
                const visionNote = images.length > 0 ? " (with vision)" : " (text-only)";
                this.statusEl.textContent = "Enhanced with " + this.taskSelect.value + " template" + visionNote;
            } else {
                this.statusEl.textContent = "Failed: " + (data.error || "Unknown error");
            }
        } catch (e) {
            this.statusEl.textContent = "Error: " + e.message;
        } finally {
            this._setEnhanceBusy(false);
        }
    }

    /** Immediately unload the selected Ollama model from VRAM. */
    async _unloadOllama() {
        const model = this.modelSelect.value;
        if (!model) { this.statusEl.textContent = "Select a model first."; return; }
        this.unloadBtn.disabled = true;
        this.unloadBtn.textContent = "Unloading...";
        this.statusEl.textContent = "Unloading " + model + " from VRAM...";
        try {
            const resp = await api.fetchApi("/bernini_studio/unload", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ollama_url: this.urlInput.value,
                    model: model,
                }),
            });
            const data = await resp.json();
            if (data.status === "unloaded") {
                this.statusEl.textContent = model + " unloaded from VRAM.";
            } else {
                this.statusEl.textContent = "Unload failed: " + (data.error || "unknown");
            }
        } catch (e) {
            this.statusEl.textContent = "Unload error: " + e.message;
        } finally {
            this.unloadBtn.disabled = false;
            this.unloadBtn.textContent = "Unload Now";
        }
    }

    /** Amber pulsing "waiting" state on the Enhance button (Looper pattern). */
    _setEnhanceBusy(busy, statusText) {
        if (busy) {
            this.enhanceBtn.disabled = true;
            this.enhanceBtn.textContent = "Enhancing...";
            this.enhanceBtn.style.background = "#d97706";
            this.enhanceBtn.style.animation = "bernini-pulse 1.4s ease-in-out infinite";
            this.enhanceBtn.style.cursor = "wait";
            this.enhanceBtn.style.opacity = "1";
            if (statusText) this.statusEl.textContent = statusText;
        } else {
            this.enhanceBtn.disabled = false;
            this.enhanceBtn.textContent = "Enhance Prompt";
            this.enhanceBtn.style.background = C.accent;
            this.enhanceBtn.style.animation = "";
            this.enhanceBtn.style.cursor = "pointer";
            this.enhanceBtn.style.opacity = "1";
        }
    }

    /* ── reference image slot grid ──────────────────────────────── */

    /**
     * Re-sync all DOM elements from their underlying widget values.
     * Called from onConfigure after ComfyUI restores saved widget values
     * (which happens AFTER onNodeCreated, so the editor DOM would
     * otherwise still show defaults from construction time).
     */
    _syncFromWidgets() {
        if (this.taskWidget && this.taskSelect) {
            this.taskSelect.value = this.taskWidget.value || "v2v";
        }
        if (this.promptWidget && this.promptArea) {
            this.promptArea.value = this.promptWidget.value || "";
        }
        if (this.negWidget && this.negArea) {
            this.negArea.value = this.negWidget.value || "";
        }
        if (this.urlWidget && this.urlInput) {
            this.urlInput.value = this.urlWidget.value || "http://127.0.0.1:11434";
        }
        if (this.apiFormatWidget && this.apiFormatSelect) {
            this.apiFormatSelect.value = this.apiFormatWidget.value || "Ollama";
        }
        if (this.autoEnhanceWidget && this.autoEnhanceCheck) {
            this.autoEnhanceCheck.checked = !!this.autoEnhanceWidget.value;
        }
        if (this.unloadOllamaWidget && this.unloadCheck) {
            this.unloadCheck.checked = !!this.unloadOllamaWidget.value;
        }
        if (this.useDefNegWidget && this.defNegCheck) {
            this.defNegCheck.checked = this.useDefNegWidget.value !== false;
        }
        // Model select: ensure saved model appears as an option even before refresh
        if (this.modelWidget && this.modelWidget.value && this.modelSelect) {
            const savedModel = this.modelWidget.value;
            let found = false;
            for (const opt of this.modelSelect.options) {
                if (opt.value === savedModel) { found = true; break; }
            }
            if (!found) {
                const o = document.createElement("option");
                o.value = savedModel;
                o.textContent = savedModel;
                this.modelSelect.appendChild(o);
            }
            this.modelSelect.value = savedModel;
        }
        // Slot grid: reload filenames from the restored widget and re-render
        this._slotFiles = this._readSlotWidget();
        this._renderSlots();
        // Augment sliders (coerce NaN/undefined to 0, never the range midpoint)
        if (this.augStrengthWidget && this.augStrengthRow) {
            const v = parseFloat(this.augStrengthWidget.value);
            this.augStrengthRow._slider.value = isNaN(v) ? 0 : v;
            this.augStrengthRow._valEl.textContent = String(this.augStrengthRow._slider.value);
        }
        if (this.augDecayWidget && this.augDecayRow) {
            const v = parseFloat(this.augDecayWidget.value);
            this.augDecayRow._slider.value = isNaN(v) ? 0 : v;
            this.augDecayRow._valEl.textContent = String(this.augDecayRow._slider.value);
        }
        if (this.augSeedWidget && this.augSeedInput) {
            this.augSeedInput.value = this.augSeedWidget.value || 0;
        }
        // Refresh hint box / placeholders for the restored task
        this._updateTaskDisplay();
        this._updateNegPlaceholder();
    }

    _readSlotWidget() {
        try {
            const v = this.slotImagesWidget ? this.slotImagesWidget.value : "[]";
            const parsed = JSON.parse(v || "[]");
            const arr = Array.isArray(parsed) ? parsed : [];
            while (arr.length < 8) arr.push(null);
            return arr.slice(0, 8);
        } catch (e) {
            return [null, null, null, null, null, null, null, null];
        }
    }

    _syncSlotWidget() {
        if (this.slotImagesWidget) {
            this.slotImagesWidget.value = JSON.stringify(this._slotFiles);
        }
        if (this.node.graph) this.node.graph.setDirtyCanvas(true, true);
    }

    _renderSlots() {
        // Count badge on the collapsed header + auto-expand if content exists
        const filled = this._slotFiles.filter(Boolean).length;
        if (this.slotCountBadge) {
            if (filled > 0) {
                this.slotCountBadge.textContent = filled + " loaded";
                this.slotCountBadge.style.display = "inline";
                if (!this._slotOpen && !this._slotAutoExpanded) {
                    // Auto-expand once when content exists (e.g. workflow load)
                    this._slotAutoExpanded = true;
                    this._slotOpen = true;
                    this.slotBody.style.display = "flex";
                    this.slotArrow.style.transform = "rotate(90deg)";
                }
            } else {
                this.slotCountBadge.style.display = "none";
            }
        }

        this.slotGrid.innerHTML = "";
        for (let i = 0; i < 8; i++) {
            const filename = this._slotFiles[i];
            const slot = el("div", {
                position: "relative", aspectRatio: "1 / 1", minHeight: "54px",
                background: C.input, border: "1px dashed " + C.borderIn,
                borderRadius: "4px", overflow: "hidden", cursor: "pointer",
                display: "flex", alignItems: "center", justifyContent: "center",
            });
            slot.dataset.slot = i;

            if (filename) {
                slot.style.border = "1px solid " + C.borderIn;
                const img = document.createElement("img");
                img.src = "/view?" + new URLSearchParams({ filename, type: "input" }).toString();
                img.style.cssText = "width:100%;height:100%;object-fit:cover;pointer-events:none;";
                slot.appendChild(img);

                // filled slots are draggable for reordering
                slot.draggable = true;
                slot.ondragstart = (e) => {
                    e.dataTransfer.setData("bernini-slot", String(i));
                    e.dataTransfer.effectAllowed = "move";
                };

                // clear button
                const clearBtn = el("div", {
                    position: "absolute", top: "2px", right: "2px",
                    width: "14px", height: "14px", lineHeight: "13px",
                    background: "rgba(0,0,0,0.7)", color: C.text,
                    borderRadius: "3px", fontSize: "10px", textAlign: "center",
                    cursor: "pointer", zIndex: "2",
                }, "\u00D7");
                clearBtn.onclick = (e) => {
                    e.stopPropagation();
                    this._slotFiles[i] = null;
                    this._syncSlotWidget();
                    this._renderSlots();
                };
                slot.appendChild(clearBtn);
            } else {
                slot.appendChild(el("span", {
                    fontSize: "9px", color: C.dim, pointerEvents: "none",
                }, "image" + i));
            }

            // label badge bottom-left on filled slots
            if (filename) {
                slot.appendChild(el("div", {
                    position: "absolute", bottom: "2px", left: "2px",
                    background: "rgba(0,0,0,0.7)", color: C.cyan,
                    fontSize: "8px", padding: "0 4px", borderRadius: "2px",
                    fontFamily: "monospace", pointerEvents: "none",
                }, "image" + i));
            }

            // click empty area -> file picker
            slot.onclick = () => {
                this.slotFileInput.onchange = async () => {
                    const file = this.slotFileInput.files[0];
                    this.slotFileInput.value = "";
                    if (file) await this._uploadToSlot(i, file);
                };
                this.slotFileInput.click();
            };

            // drop target: OS file drop OR slot-to-slot move
            slot.ondragover = (e) => {
                e.preventDefault();
                slot.style.borderColor = C.accent;
            };
            slot.ondragleave = () => {
                slot.style.borderColor = filename ? C.borderIn : C.borderIn;
            };
            slot.ondrop = async (e) => {
                e.preventDefault();
                e.stopPropagation();
                slot.style.borderColor = C.borderIn;
                const fromSlot = e.dataTransfer.getData("bernini-slot");
                if (fromSlot !== "") {
                    // reorder: swap source and target slots
                    const from = parseInt(fromSlot, 10);
                    if (!isNaN(from) && from !== i) {
                        const tmp = this._slotFiles[i];
                        this._slotFiles[i] = this._slotFiles[from];
                        this._slotFiles[from] = tmp;
                        this._syncSlotWidget();
                        this._renderSlots();
                    }
                    return;
                }
                // OS file drop
                const file = e.dataTransfer.files && e.dataTransfer.files[0];
                if (file && file.type.startsWith("image/")) {
                    await this._uploadToSlot(i, file);
                }
            };

            this.slotGrid.appendChild(slot);
        }
        // resize node to fit
        requestAnimationFrame(() => {
            if (this.node.computeSize && this.node.size) {
                this.node.size[1] = this.node.computeSize()[1];
            }
            if (this.node.graph) this.node.graph.setDirtyCanvas(true, true);
        });
    }

    async _uploadToSlot(slotIndex, file) {
        try {
            const formData = new FormData();
            formData.append("image", file);
            formData.append("type", "input");
            formData.append("overwrite", "false");
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body: formData,
            });
            if (!resp.ok) {
                this.statusEl.textContent = "Upload failed: HTTP " + resp.status;
                return;
            }
            const data = await resp.json();
            // ComfyUI returns {name, subfolder, type}; may rename on collision
            let filename = data.name || file.name;
            if (data.subfolder) filename = data.subfolder + "/" + filename;
            this._slotFiles[slotIndex] = filename;
            this._syncSlotWidget();
            this._renderSlots();
            console.log("[BerniniStudio] Uploaded to slot", slotIndex, ":", filename);
        } catch (e) {
            console.warn("[BerniniStudio] Slot upload failed:", e);
            this.statusEl.textContent = "Upload error: " + e.message;
        }
    }

    /**
     * Extract frames from the connected source video for LLM vision.
     * Traces source_video input to find VHS_LoadVideo (or similar),
     * gets the video filename, and calls the server to extract frames via ffmpeg.
     */
    async _collectSourceVideoFrames() {
        const frames = [];
        const graph = app.graph || this.node.graph;
        if (!graph) return frames;

        const input = (this.node.inputs || []).find(inp => inp.name === "source_video");
        if (!input || input.link == null) return frames;

        const link = graph.links[input.link];
        if (!link) return frames;

        // Walk backwards to find a node with a "video" widget (VHS_LoadVideo pattern)
        const visited = new Set();
        let current = graph.getNodeById(link.origin_id);
        let videoFilename = null;
        let subfolder = "";

        while (current && !visited.has(current.id)) {
            visited.add(current.id);
            const vidWidget = (current.widgets || []).find(w => w.name === "video");
            if (vidWidget && vidWidget.value) {
                if (typeof vidWidget.value === "object") {
                    videoFilename = vidWidget.value.filename || "";
                    subfolder = vidWidget.value.subfolder || "";
                } else {
                    videoFilename = vidWidget.value;
                }
                break;
            }
            // Also check for "image" widget (LoadImage used as single frame source)
            const imgWidget = (current.widgets || []).find(w => w.name === "image");
            if (imgWidget && imgWidget.value) {
                try {
                    let fn = typeof imgWidget.value === "object" ? imgWidget.value.filename : imgWidget.value;
                    const resp = await api.fetchApi("/view?" + new URLSearchParams({ filename: fn, type: "input" }).toString());
                    if (resp.ok) {
                        const blob = await resp.blob();
                        frames.push(await this._blobToBase64(blob));
                    }
                } catch (e) {}
                return frames;
            }
            // Follow upstream IMAGE input
            const imgInput = (current.inputs || []).find(inp =>
                inp.link != null && (inp.type === "IMAGE" || inp.name === "image")
            );
            if (!imgInput) break;
            const upLink = graph.links[imgInput.link];
            if (!upLink) break;
            current = graph.getNodeById(upLink.origin_id);
        }

        if (!videoFilename) return frames;
        console.log("[BerniniStudio] Extracting frames from:", videoFilename);

        try {
            const resp = await api.fetchApi("/bernini_studio/extract_frames", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: videoFilename, subfolder, num_frames: 3 }),
            });
            const data = await resp.json();
            if (data.frames && data.frames.length > 0) {
                console.log("[BerniniStudio] Got", data.frames.length, "source video frames");
                return data.frames;
            }
        } catch (e) {
            console.warn("[BerniniStudio] Frame extraction failed:", e);
        }
        return frames;
    }

    /**
     * Collect base64-encoded reference images from connected nodes.
     * Traces the graph links from this node's reference_image_N inputs
     * backwards through any intermediate nodes (Resize, Preview, etc.)
     * until it finds a LoadImage node, then fetches the image via
     * ComfyUI's /view endpoint.
     */
    async _collectReferenceImages() {
        const images = [];
        const graph = app.graph || this.node.graph;
        if (!graph) { console.warn("[BerniniStudio] No graph available"); return images; }

        const inputNames = (this.node.inputs || []).map(inp => inp.name);
        console.log("[BerniniStudio] Node inputs:", inputNames.join(", "));

        for (let i = 0; i < 8; i++) {
            const inputName = "image" + i;
            const input = (this.node.inputs || []).find(inp => inp.name === inputName);
            if (!input || input.link == null) {
                // No jack wired -- fall back to built-in editor slot (same priority as Python)
                const slotFile = this._slotFiles && this._slotFiles[i];
                if (slotFile) {
                    try {
                        const resp = await api.fetchApi("/view?" + new URLSearchParams({
                            filename: slotFile, type: "input",
                        }).toString());
                        if (resp.ok) {
                            const blob = await resp.blob();
                            images.push(await this._blobToBase64(blob));
                            console.log("[BerniniStudio] image" + i, "from editor slot:", slotFile);
                        }
                    } catch (e) {
                        console.warn("[BerniniStudio] Slot fetch failed for image" + i, e);
                    }
                }
                continue;
            }
            console.log("[BerniniStudio]", inputName, "linked (link id:", input.link, ")");

            const link = graph.links[input.link];
            if (!link) { console.warn("[BerniniStudio]", inputName, "link not found in graph"); continue; }

            const sourceNode = graph.getNodeById(link.origin_id);
            console.log("[BerniniStudio]", inputName, "-> node", link.origin_id,
                        sourceNode ? sourceNode.type || sourceNode.comfyClass : "(null)");

            const imgWidget = this._traceToImageWidget(graph, sourceNode);
            if (!imgWidget || !imgWidget.value) {
                console.warn("[BerniniStudio]", inputName, "traced but no image widget found");
                continue;
            }
            console.log("[BerniniStudio]", inputName, "found file:", imgWidget.value);

            try {
                let filename, subfolder = "", type = "input";
                if (typeof imgWidget.value === "object") {
                    filename = imgWidget.value.filename || "";
                    subfolder = imgWidget.value.subfolder || "";
                    type = imgWidget.value.type || "input";
                } else {
                    filename = imgWidget.value;
                }
                if (!filename) continue;

                let viewUrl = "/view?" + new URLSearchParams({ filename, type }).toString();
                if (subfolder) viewUrl += "&subfolder=" + encodeURIComponent(subfolder);

                const resp = await api.fetchApi(viewUrl);
                if (!resp.ok) continue;
                const blob = await resp.blob();
                const b64 = await this._blobToBase64(blob);
                images.push(b64);
            } catch (e) {
                console.warn("[BerniniStudio] Failed to fetch image from", inputName, e);
            }
        }
        return images;
    }

    /**
     * Walk backwards through the graph from a given node, following IMAGE
     * inputs, until we find a node with an "image" widget (LoadImage, etc).
     * Handles chains like: LoadImage -> Resize -> Preview -> BerniniStudio.
     * Returns the widget object or null.
     */
    _traceToImageWidget(graph, node) {
        const visited = new Set();
        let current = node;
        while (current && !visited.has(current.id)) {
            visited.add(current.id);

            // Check if this node has an "image" file widget (LoadImage pattern)
            const imgWidget = (current.widgets || []).find(w => w.name === "image");
            if (imgWidget && imgWidget.value) return imgWidget;

            // No widget found -- look for an IMAGE-type input link and follow it upstream.
            // Try common input names first, then fall back to any linked input.
            const imgInput = (current.inputs || []).find(inp => {
                if (inp.link == null) return false;
                if (inp.type === "IMAGE") return true;
                if (inp.name === "image" || inp.name === "images") return true;
                return false;
            });
            if (!imgInput) return null;

            const upLink = graph.links[imgInput.link];
            if (!upLink) return null;

            current = graph.getNodeById(upLink.origin_id);
        }
        return null;
    }

    _blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                // reader.result is "data:image/...;base64,XXXXX"
                // Strip the prefix to get raw base64
                const result = reader.result;
                const idx = result.indexOf(",");
                resolve(idx >= 0 ? result.substring(idx + 1) : result);
            };
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(blob);
        });
    }

    _installVisibilityWatchdog() {
        let ticks = 0;
        const tick = () => {
            for (const name of HIDDEN_WIDGETS) {
                const w = findWidget(this.node, name);
                if (w && !w.__bernini_hidden) hideWidget(w);
            }
            ticks++;
            if (ticks < 30) requestAnimationFrame(tick);
        };
        tick();
    }
}

/* ── Extension registration ─────────────────────────────────────── */

app.registerExtension({
    name: "BerniniStudio.Editor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BerniniStudio") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            try {
                for (const name of HIDDEN_WIDGETS) hideWidget(findWidget(this, name));

                const mount = document.createElement("div");
                mount.style.cssText = "width:100%;box-sizing:border-box;";
                const getEditorHeight = () => {
                    const ed = mount.firstElementChild;
                    return ed ? ed.scrollHeight : 400;
                };

                const editorWidget = this.addDOMWidget("bernini_editor", "div", mount, {
                    serialize: false, hideOnZoom: false, getHeight: getEditorHeight,
                });
                if (editorWidget) {
                    editorWidget.computeSize = function (width) {
                        return [width, getEditorHeight()];
                    };
                }

                new BerniniEditor(this, mount);

                requestAnimationFrame(() => {
                    if (this.computeSize && this.size) this.size[1] = this.computeSize()[1];
                    if (this.graph) this.graph.setDirtyCanvas(true, true);
                });
            } catch (err) {
                console.error("[BerniniStudio] Editor setup error:", err);
            }

            return r;
        };

        // onConfigure fires AFTER ComfyUI restores saved widget values from
        // the workflow JSON. At that point the underlying widgets hold their
        // saved values, but the editor DOM still shows defaults from
        // construction time. Re-sync everything.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this._berniniEditor) {
                requestAnimationFrame(() => {
                    this._berniniEditor._syncFromWidgets();
                    if (this.computeSize && this.size) this.size[1] = this.computeSize()[1];
                    if (this.graph) this.graph.setDirtyCanvas(true, true);
                });
            }
            return r;
        };
    },
});
