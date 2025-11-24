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
