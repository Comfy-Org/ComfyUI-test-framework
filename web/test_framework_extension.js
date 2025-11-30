import { app } from "../../scripts/app.js";

/**
 * ComfyUI Test Framework Extension
 *
 * Automatically adds API format to workflows that contain TestDefinition nodes
 * when they are exported. This allows test workflows to be executed directly
 * via the API without manual conversion.
 */
app.registerExtension({
    name: "comfyui.test.framework.auto.save.api",

    async init(app) {
        // Hook into app.graphToPrompt to inject API format into workflow
        const original_graphToPrompt = app.graphToPrompt.bind(app);

        app.graphToPrompt = async function(graph) {
            // Call original function to get both workflow and API formats
            const result = await original_graphToPrompt(graph);

            // Check if workflow contains TestDefinition node
            const hasTestDefinition = result.workflow.nodes?.some(
                node => node.type === "TestDefinition"
            );

            if (hasTestDefinition) {
                // Add API format to the workflow object
                result.workflow.api = result.output;
                console.log("[Test Framework] Added API format to workflow");
            }

            return result;
        };
    }
});

/**
 * AssertImageMatch Accept Button Extension
 *
 * Adds an "Accept Hash" button to AssertImageMatch nodes that copies the
 * calculated perceptual hash into the perceptual_hash input field.
 */
app.registerExtension({
    name: "comfyui.test.framework.image.match.accept",

    async nodeCreated(node) {
        if (node.comfyClass === "AssertImageMatch") {
            // Add "Accept Hash" button
            const acceptButton = node.addWidget(
                "button",
                "Accept Hash",
                null,
                () => {
                    // Find the perceptual_hash widget
                    const hashWidget = node.widgets.find(w => w.name === "perceptual_hash");

                    if (!hashWidget) {
                        console.error("[Test Framework] Could not find perceptual_hash widget");
                        return;
                    }

                    // Get the last execution output for this node
                    const nodeOutputs = node.imgs;

                    // Look for the hash in the node's last output
                    // The hash is sent via PreviewText in the execute method
                    if (node.widgets_values && node.widgets_values.length > 0) {
                        // Try to extract hash from cached execution data
                        const lastHash = node._lastCalculatedHash;
                        if (lastHash) {
                            hashWidget.value = lastHash;
                            console.log("[Test Framework] Accepted hash:", lastHash);
                            return;
                        }
                    }

                    // If we can't find cached hash, try to parse from text output
                    // This is a fallback - ideally we'd capture it from execution
                    console.warn("[Test Framework] No calculated hash found. Please run the node first.");
                    alert("Please execute the node first to calculate the hash.");
                },
                { serialize: false }
            );

            // Listen for progress_text events to capture the calculated hash
            // The hash is sent via send_progress_text() which triggers a progress_text WebSocket event
            app.api.addEventListener('progress_text', (e) => {
                // Filter for events matching this node's ID
                if (e.detail.nodeId === node.id.toString()) {
                    const text = e.detail.text;

                    // Try to extract a 64-character binary string (dhash format)
                    const hashMatch = text.match(/[01]{64}/);
                    if (hashMatch) {
                        node._lastCalculatedHash = hashMatch[0];
                        console.log("[Test Framework] Captured hash:", node._lastCalculatedHash);
                    } else {
                        // If it's a pure hash string without formatting
                        const pureHash = text.trim();
                        if (/^[01]+$/.test(pureHash)) {
                            node._lastCalculatedHash = pureHash;
                            console.log("[Test Framework] Captured hash:", node._lastCalculatedHash);
                        }
                    }
                }
            });
        }
    }
});
