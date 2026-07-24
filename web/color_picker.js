import { app } from "../../scripts/app.js";

function normalizeHexColor(value) {
    const text = String(value ?? "").trim();
    const short = /^#?([0-9a-fA-F]{3})$/.exec(text);
    if (short) {
        return `#${[...short[1]].map((character) => character.repeat(2)).join("")}`.toLowerCase();
    }
    const full = /^#?([0-9a-fA-F]{6})$/.exec(text);
    return full ? `#${full[1].toLowerCase()}` : "#ffd9ad";
}

app.registerExtension({
    name: "RTXIllustrationEnhancer.ColorPicker",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RTXIllustrationEnhancer") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            const colorWidget = this.widgets?.find((widget) => widget.name === "light_color");
            if (!colorWidget) {
                return result;
            }

            const pickerWidget = this.addWidget(
                "button",
                "🎨 choose light color",
                null,
                () => {
                    const picker = document.createElement("input");
                    picker.type = "color";
                    picker.value = normalizeHexColor(colorWidget.value);
                    picker.setAttribute("aria-label", "Choose light color");
                    Object.assign(picker.style, {
                        position: "fixed",
                        left: "-100px",
                        top: "-100px",
                        width: "1px",
                        height: "1px",
                        opacity: "0",
                        pointerEvents: "none",
                    });

                    const updateColor = () => {
                        const value = picker.value.toLowerCase();
                        colorWidget.value = value;
                        colorWidget.callback?.(value);
                        pickerWidget.label = `🎨 ${value}`;
                        this.setDirtyCanvas?.(true, true);
                    };
                    const cleanup = () => picker.remove();

                    picker.addEventListener("input", updateColor);
                    picker.addEventListener("change", () => {
                        updateColor();
                        cleanup();
                    }, { once: true });
                    picker.addEventListener("cancel", cleanup, { once: true });
                    document.body.appendChild(picker);
                    picker.click();
                },
                { serialize: false },
            );
            pickerWidget.label = `🎨 ${normalizeHexColor(colorWidget.value)}`;
            return result;
        };
    },
});
