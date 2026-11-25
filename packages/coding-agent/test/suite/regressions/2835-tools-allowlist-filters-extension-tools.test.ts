import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, getCurrentTools } from "@earendil-works/pi-ai";
import { getModel } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DefaultResourceLoader } from "../../../src/core/resource-loader.ts";
import { createAgentSession } from "../../../src/core/sdk.ts";
import { SessionManager } from "../../../src/core/session-manager.ts";
import { SettingsManager } from "../../../src/core/settings-manager.ts";
import type { ToolExposure } from "../../../src/index.ts";
import { createHarness } from "../harness.ts";

describe("regression #2835: tool allowlists filter extension tools", () => {
	let tempDir: string;
	let agentDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-tools-filter-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		agentDir = join(tempDir, "agent");
		mkdirSync(agentDir, { recursive: true });
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	async function createSession(allowedToolNames?: string[]) {
		const settingsManager = SettingsManager.create(tempDir, agentDir);
		const sessionManager = SessionManager.inMemory(tempDir);
		const resourceLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager,
			extensionFactories: [
				(pi) => {
					pi.on("session_start", () => {
						pi.registerTool({
							name: "dynamic_tool",
							label: "Dynamic Tool",
							description: "Tool registered from session_start",
							promptSnippet: "Run dynamic test behavior",
							parameters: Type.Object({}),
							execute: async () => ({
								content: [{ type: "text", text: "ok" }],
								details: {},
							}),
						});
					});
				},
			],
		});
		await resourceLoader.reload();

		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			model: getModel("anthropic", "claude-sonnet-4-5")!,
			settingsManager,
			sessionManager,
			resourceLoader,
			tools: allowedToolNames,
		});
		await session.bindExtensions({});
		return session;
	}

	it("allows only explicitly listed built-in and extension tools", async () => {
		const session = await createSession(["read", "dynamic_tool"]);

		expect(
			session
				.getAllTools()
				.map((tool) => tool.name)
				.sort(),
		).toEqual(["dynamic_tool", "read"]);
		expect(session.getActiveToolNames().sort()).toEqual(["dynamic_tool", "read"]);
		expect(session.systemPrompt).toContain("- read: Read file contents");
		expect(session.systemPrompt).toContain("- dynamic_tool: Run dynamic test behavior");
		expect(session.systemPrompt).not.toContain("- bash:");
		expect(session.systemPrompt).not.toContain("- edit:");
		session.dispose();
	});

	it("disables all tools when the allowlist is empty", async () => {
		const session = await createSession([]);

		expect(session.getAllTools()).toEqual([]);
		expect(session.getActiveToolNames()).toEqual([]);
		expect(session.systemPrompt).toContain("<tools>\n(none)\n");
		expect(session.systemPrompt).not.toContain("dynamic_tool");
		session.dispose();
	});

	// Regression #12399
	it.each([
		{ selection: "read only", activeTools: ["read"] },
		{ selection: "empty", activeTools: [] },
	])("preserves a $selection selection after registering an unrelated tool", async ({ activeTools }) => {
		const harness = await createHarness({
			allowedToolNames: ["read", "grep"],
			extensionFactories: [
				(pi) => {
					pi.on("session_start", () => {
						pi.setActiveTools(activeTools);
						pi.registerTool({
							name: "unrelated",
							label: "Unrelated",
							description: "Tool outside the allowlist",
							parameters: Type.Object({}),
							execute: async () => ({ content: [], details: {} }),
						});
					});
				},
			],
		});
		try {
			await harness.session.bindExtensions({});
			const activeToolNames = harness.session.getActiveToolNames();
			const providerToolNames: string[][] = [];
			harness.setResponses([
				(context) => {
					providerToolNames.push(getCurrentTools(context.messages).map((tool) => tool.name));
					return fauxAssistantMessage("done");
				},
			]);
			await harness.session.prompt("start");

			expect(
				harness.session
					.getAllTools()
					.map((tool) => tool.name)
					.sort(),
			).toEqual(["grep", "read"]);
			expect(activeToolNames).toEqual(activeTools);
			expect(providerToolNames).toEqual([activeTools]);
		} finally {
			harness.cleanup();
		}
	});

	// Regression #12399
	it.each([
		{ pattern: "late_tool", exposure: "direct" },
		{ pattern: "late_*", exposure: "model-only" },
	] as const)("activates newly declarable $exposure tools matching $pattern", async ({ pattern, exposure }) => {
		let registerLateTool!: (exposure: ToolExposure, description: string) => void;
		const harness = await createHarness({
			allowedToolNames: ["read", "grep", pattern],
			extensionFactories: [
				(pi) => {
					registerLateTool = (exposure, description) =>
						pi.registerTool({
							name: "late_tool",
							label: "Late Tool",
							description,
							exposure,
							defaultActive: false,
							parameters: Type.Object({}),
							execute: async () => ({ content: [], details: {} }),
						});
				},
			],
		});
		try {
			const { session } = harness;
			session.setActiveToolsByName(["read"]);

			registerLateTool(exposure, "New declaration");
			expect(session.getActiveToolNames()).toEqual(["read", "late_tool"]);

			registerLateTool(exposure, "Updated active declaration");
			expect(session.getActiveToolNames()).toEqual(["read", "late_tool"]);
			session.setActiveToolsByName(["read"]);
			registerLateTool(exposure, "Updated inactive declaration");
			expect(session.getToolDefinition("late_tool")?.description).toBe("Updated inactive declaration");
			expect(session.getActiveToolNames()).toEqual(["read"]);

			registerLateTool("hidden", "Hidden declaration");
			expect(session.getActiveToolNames()).toEqual(["read"]);
			registerLateTool(exposure, "Visible declaration");
			expect(session.getActiveToolNames()).toEqual(["read", "late_tool"]);
		} finally {
			harness.cleanup();
		}
	});
});
