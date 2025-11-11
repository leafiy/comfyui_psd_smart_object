import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PSD_NODE_CLASSES = ["PSDFileUploader", "PSDMockupEmbedder", "PSDSmartObjectInspector"];
const ACCEPT_EXTENSIONS = [".psd", ".psb"];

function uploadPSDFile(file, node, onProgress) {
    return new Promise((resolve, reject) => {
        try {
            const formData = new FormData();
            formData.append("image", file);
            const request = new XMLHttpRequest();
            request.upload.onprogress = (event) => {
                if (event.lengthComputable) {
                    onProgress?.(event.loaded / event.total);
                }
            };
            request.onload = () => {
                node.progress = undefined;
                if (request.status !== 200) {
                    const message = request.responseText || request.statusText;
                    app.ui?.alert?.(`PSD upload failed: ${message}`);
                    reject(new Error(message));
                    return;
                }
                try {
                    const payload = JSON.parse(request.responseText);
                    resolve(payload?.name || file.name);
                } catch (error) {
                    reject(error);
                }
            };
            request.onerror = () => {
                node.progress = undefined;
                reject(new Error("Network error"));
            };
            request.open("POST", api.apiURL("/upload/image"), true);
            request.send(formData);
        } catch (error) {
            node.progress = undefined;
            app.ui?.alert?.(`PSD upload failed: ${error.message || error}`);
            reject(error);
        }
    });
}

function pushValue(widget, value) {
    if (!widget) {
        return;
    }
    const values = widget.options?.values;
    if (Array.isArray(values)) {
        if (values.length === 1 && typeof values[0] === "string" && values[0].startsWith("<")) {
            values.splice(0, 1);
        }
        if (!values.includes(value)) {
            values.push(value);
        }
    }
    widget.value = value;
    if (widget.callback) {
        widget.callback(value);
    }
}

function attachUploadControls(node) {
    if (!PSD_NODE_CLASSES.includes(node.comfyClass) || node.__psdUploadAttached) {
        return;
    }
    const widget = node.widgets?.find?.((w) => w.name === "psd_file");
    if (!widget) {
        return;
    }

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ACCEPT_EXTENSIONS.join(",");
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    const cleanup = () => {
        fileInput.remove();
        node.__psdUploadAttached = false;
    };

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        cleanup();
        if (originalOnRemoved) {
            return originalOnRemoved.apply(this, arguments);
        }
    };

    const acceptsExt = (filename) => {
        const lower = filename?.toLowerCase?.();
        return !!lower && ACCEPT_EXTENSIONS.some((ext) => lower.endsWith(ext));
    };

    const handleFile = async (file) => {
        if (!file || !acceptsExt(file.name)) {
            app.ui?.alert?.("Only PSD/PSB files are supported.");
            return false;
        }
        try {
            node.progress = 0.01;
            const uploadedName = await uploadPSDFile(file, node, (ratio) => {
                node.progress = Math.min(0.99, Math.max(0.01, ratio));
            });
            pushValue(widget, uploadedName);
            node.graph?.setDirtyCanvas(true, true);
            return true;
        } catch (error) {
            return false;
        } finally {
            node.progress = undefined;
        }
    };

    fileInput.onchange = async () => {
        const file = fileInput.files?.[0];
        fileInput.value = "";
        await handleFile(file);
    };

    const button = node.addWidget("button", "choose PSD to upload", "upload_psd", () => {
        app.canvas?.node_widget && (app.canvas.node_widget = null);
        fileInput.click();
    });
    button.options = button.options || {};
    button.options.serialize = false;

    const previousDragOver = node.onDragOver;
    node.onDragOver = function (event) {
        if (event?.dataTransfer?.types?.includes?.("Files")) {
            return true;
        }
        return previousDragOver ? previousDragOver.call(this, event) : false;
    };

    const previousDragDrop = node.onDragDrop;
    node.onDragDrop = async function (event) {
        if (event?.dataTransfer?.files?.length) {
            return await handleFile(event.dataTransfer.files[0]);
        }
        if (previousDragDrop) {
            return previousDragDrop.call(this, event);
        }
        return false;
    };

    node.__psdUploadAttached = true;
}

app.registerExtension({
    name: "PSDMockup.UploadControls",
    nodeCreated(node) {
        attachUploadControls(node);
    },
});
