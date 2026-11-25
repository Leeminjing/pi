// Save a cancellation request and wait for remote confirmation across a SQLite close and reopen.
// Run from packages/durable:
//   node --conditions=source --experimental-strip-types test/examples/48-durable-cancellation.ts

import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { createModels } from "@earendil-works/pi-ai/models";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { obj, str } from "@earendil-works/pi-schema";
import {
	createRegistry,
	defineExtension,
	defineSignal,
	defineTool,
	Harness,
	type TaskId,
	ToolResultEntry,
} from "../../src/index.ts";
import { openNodeSqliteStorage } from "../../src/storage/sqlite/node.ts";

const context = BACKGROUND_CONTEXT;
const BuildFinished = defineSignal({ name: "example.build-finished", version: 1, schema: str() });
const BuildStopped = defineSignal({
	name: "example.build-stopped",
	version: 1,
	delivery: "cancellation",
	schema: str(),
});
const parameters = obj({ ref: str() });

async function run(path: string, confirmation: "early" | "after reopen" | "timeout"): Promise<void> {
	const requestStarted = Promise.withResolvers<string>();
	const acceptRequest = Promise.withResolvers<void>();
	// Stands in for an external service that remembers its requests independently of the Harness.
	const stopRequests = new Map<string, string>();
	const jobs = new Map<string, "running" | "stopped">();
	const acknowledge = async (harness: Harness, id: TaskId, requestId: string) => {
		const jobId = stopRequests.get(requestId);
		assert(jobId);
		jobs.set(jobId, "stopped");
		assert.equal(await harness.signal(id, BuildStopped, requestId, jobId, context), "delivered");
	};
	const wakes: Array<"signal" | "expired" | "time" | undefined> = [];
	let executions = 0;
	let now = 1_000;
	const until = now + 60_000;
	const build = defineTool<typeof parameters, null, { jobId: string }, { requestId: string }>({
		name: "build",
		description: "Start a remote build",
		parameters,
		async execute({ ref }) {
			executions++;
			const jobId = `job:${ref}`;
			jobs.set(jobId, "running");
			return { kind: "sleep", checkpoint: { jobId }, signals: [BuildFinished.key("done")] };
		},
		async cancel(_args, api, ctx) {
			assert(api.checkpoint);
			const { jobId } = api.checkpoint;
			// Temporary cleanup uses a one-shot request. Durable cancellation can save a continuation.
			if (api.reason !== "abort") {
				stopRequests.set(`stop:${jobId}`, jobId);
				return;
			}
			wakes.push(api.wake?.reason);
			if (api.cancelCheckpoint === undefined) {
				const requestId = await api.memo("stop-request", `stop:${jobId}`, ctx);
				stopRequests.set(requestId, jobId);
				requestStarted.resolve(requestId);
				if (confirmation === "early") await acceptRequest.promise;
				return { kind: "sleep", checkpoint: { requestId }, signals: [BuildStopped.key(requestId)], until };
			}
			const confirmed = await api.readSignal(BuildStopped, api.cancelCheckpoint.requestId, ctx);
			if (confirmed?.kind === "value") {
				assert.equal(confirmed.value, jobId);
				api.diagnostic({ severity: "info", code: "cancel_confirmed", message: `${jobId} stopped` });
			} else {
				assert.equal(api.wake?.reason, "expired");
				api.diagnostic({
					severity: "warn",
					code: "cancel_unconfirmed",
					message: `${jobId} did not confirm the stop`,
				});
			}
		},
	});
	const faux = fauxProvider();
	faux.setResponses([
		fauxAssistantMessage([fauxToolCall("build", { ref: "main" }, { id: "build-1" })], { stopReason: "toolUse" }),
	]);
	const models = createModels();
	models.setProvider(faux.provider);
	const registry = createRegistry();
	registry.install(defineExtension({ name: "remote-build", tools: [build] }));
	const open = async () =>
		Harness.open(
			await openNodeSqliteStorage(path),
			{
				models,
				registry,
				now: () => now,
				onWake: () => {},
			},
			context,
		);

	// Start the job and let the tool persist its ordinary wait.
	let harness = await open();
	let root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
	await root.submit({ type: "input", content: "Build main" }, context);
	await eventually(() => !harness.wake().active);
	const task = (await harness.inspect(context)).tasks.find(({ record }) => record.kind === "pi.tool")!;
	const id = task.record.id;
	assert.equal(task.record.state.status, "sleeping");

	// Observe the saved wait independently of the Stop call, which waits for the conversation to become idle.
	const stopped = Promise.allSettled([root.abort(context)]);
	const requestId = await requestStarted.promise;
	if (confirmation === "early") {
		await acknowledge(harness, id, requestId);
		acceptRequest.resolve();
		assert.equal((await stopped)[0].status, "fulfilled");
	} else {
		await eventually(() => !harness.wake().active && harness.wake().at === until);
		const saved = await harness.getTask(id, context);
		assert.equal(saved?.state.status, "sleeping");
		assert.equal(saved.state.mode, "abort");
		console.log(`Task ${id} waiting for ${requestId}`, harness.wake());
		await harness.close(context);
		assert.equal((await stopped)[0].status, "rejected");

		// Reopen the same database. The task and its deadline are unchanged.
		harness = await open();
		root = await harness.root(context);
		harness.resume();
		assert.deepEqual(await harness.getTask(id, context), saved);
		assert.deepEqual(harness.wake(), { active: false, at: until });
		assert.deepEqual(wakes, [undefined]);
		if (confirmation === "after reopen") {
			await acknowledge(harness, id, requestId);
		} else {
			now = until;
			harness.resume();
		}
	}

	const receipt = await harness.waitForTask(id, context);
	await root.waitForIdle(context);
	const entries = await root.entries({}, 20, undefined, context);
	const result = entries.items.find((entry) => ToolResultEntry.is(entry));
	const code = confirmation === "timeout" ? "cancel_unconfirmed" : "cancel_confirmed";
	assert.equal(receipt.state.outcome.status, "aborted");
	assert.equal(executions, 1);
	assert.equal(stopRequests.size, 1);
	assert.equal(jobs.get("job:main"), confirmation === "timeout" ? "running" : "stopped");
	assert.deepEqual(wakes, [undefined, confirmation === "timeout" ? "expired" : "signal"]);
	assert(result?.data.diagnostics.some((diagnostic) => diagnostic.code === code));
	console.log(`Task ${id} ${confirmation}: ${wakes.at(-1)}, ${code}`);
	await harness.close(context);
}

async function eventually(predicate: () => boolean): Promise<void> {
	for (let i = 0; i < 200; i++) {
		if (predicate()) return;
		await new Promise((resolve) => setTimeout(resolve, 5));
	}
	throw new Error("The Harness did not become inactive");
}

const directory = await mkdtemp(join(tmpdir(), "pi-durable-cancellation-"));
try {
	for (const confirmation of ["early", "after reopen", "timeout"] as const) {
		await run(join(directory, `${confirmation}.sqlite`), confirmation);
	}
} finally {
	await rm(directory, { recursive: true, force: true });
}
