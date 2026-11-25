// Commit ordering, recovery and ownership of durable cancellation waits (#12090).
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import {
	type ConversationId,
	defineExtension,
	defineSignal,
	defineTask,
	defineTool,
	type Harness,
	LiveDoc,
	type Task,
	type TaskId,
} from "@earendil-works/pi-durable";
import { obj, str } from "@earendil-works/pi-schema";
import { describe, expect, it } from "vitest";
import { chatSetup, openChat } from "./chat-support.ts";
import { addTask, addTool } from "./harness-support.ts";
import { ControlledStorage, context, flush } from "./session-support.ts";
import { aborted, abortedWith, completed, deferred, eventually, openTasks, settled } from "./task-support.ts";

const Confirm = defineSignal({ name: "test.confirm-stop", version: 1, schema: str(), delivery: "cancellation" });
const OWNED_BY_CONVERSATION = { ownership: { kind: "conversation" } } as const;
const noTimer = { onWake: () => {} };

async function start<S extends { phase: string }, R>(harness: Harness, task: Task<null, S, R, object>) {
	return (await harness.root(context)).commit((tx) => tx.createTask(task, null, OWNED_BY_CONVERSATION), context);
}

// #12090
describe("cancellation commit ordering", () => {
	it.each(["completion", "stop"] as const)("keeps the outcome of the first durable commit: %s", async (winner) => {
		const storage = new ControlledStorage();
		const started = deferred();
		const finish = deferred();
		const attempted = deferred<PromiseSettledResult<void>>();
		const cancelled: TaskId[] = [];
		const Task = defineTask<null, { phase: "work" }, string>({
			name: "test.stop-race",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: async (_task, runtime) => {
					started.resolve();
					await finish.promise;
					const [outcome] = await Promise.allSettled([runtime.commit(() => completed("successful"), context)]);
					attempted.resolve(outcome);
				},
			},
			abort: (task, runtime, ctx) => {
				cancelled.push(task.id);
				return runtime.commit(() => abortedWith("stopped"), ctx);
			},
		});
		const { harness } = await openTasks(storage, [Task]);
		const id = await start(harness, Task);
		harness.resume();
		await started.promise;
		const held = storage.holdCommits();
		if (winner === "completion") finish.resolve();
		const stop = winner === "stop" ? harness.abortTask(id, context) : undefined;
		await held.entered;
		const mark = stop ?? harness.abortTask(id, context);
		if (winner === "stop") finish.resolve();
		held.release();
		expect(await mark).toBe(winner === "completion" ? "terminal" : "marked");
		const receipt = await harness.waitForTask(id, context);
		if (winner === "completion") {
			expect(receipt.state.outcome).toEqual(completed("successful").outcome);
			expect(await attempted.promise).toEqual({ status: "fulfilled", value: undefined });
			expect(cancelled).toEqual([]);
		} else {
			expect(receipt.state.outcome).toEqual(abortedWith("stopped").outcome);
			expect(await attempted.promise).toMatchObject({
				status: "rejected",
				reason: new Error(`Task ${id} has a durable abort mark`),
			});
			expect(cancelled).toEqual([id]);
		}
		await harness.close(context);
	});

	it("keeps a completing parent's outcome while cancellation sleeps across three generations of ownership", async () => {
		const storage = new ControlledStorage();
		const ready = deferred<TaskId>();
		const parentFinishes = deferred();
		const order: string[] = [];
		const Leaf = defineTask<null, { phase: "park" }, string>({
			name: "test.stop-leaf",
			version: 1,
			initial: () => ({ phase: "park" }),
			phases: { park: (_task, runtime) => aborted(runtime.signal) },
			abort: async (task, runtime, ctx) => {
				if (task.state.wake === undefined) {
					await runtime.commit(
						() => ({
							status: "sleeping",
							checkpoint: task.state.checkpoint,
							signals: [Confirm.key("leaf")],
						}),
						ctx,
					);
					return;
				}
				order.push("leaf");
				await runtime.commit(() => abortedWith("leaf stopped"), ctx);
			},
		});
		const Middle = defineTask<null, { phase: "spawn" } | { phase: "wait" }, string>({
			name: "test.stop-middle",
			version: 1,
			initial: () => ({ phase: "spawn" }),
			phases: {
				spawn: async (task, runtime, ctx) => {
					let leaf: TaskId | undefined;
					await runtime.commit(async (tx) => {
						leaf = await tx.createTask(Leaf, null, { ownership: { kind: "task", taskId: task.id } });
						return { status: "running", checkpoint: { phase: "wait" } };
					}, ctx);
					ready.resolve(leaf!);
				},
				wait: (_task, runtime) => aborted(runtime.signal),
			},
			abort: async (_task, runtime, ctx) => {
				order.push("middle");
				await runtime.commit(() => abortedWith("middle stopped"), ctx);
			},
		});
		const Parent = defineTask<null, { phase: "spawn" }, string>({
			name: "test.stop-parent",
			version: 1,
			initial: () => ({ phase: "spawn" }),
			phases: {
				spawn: async (task, runtime, ctx) => {
					await runtime.commit(async (tx) => {
						await tx.createTask(Middle, null, { ownership: { kind: "task", taskId: task.id } });
						return undefined;
					}, ctx);
					await parentFinishes.promise;
					await runtime.commit(() => completed("saved result"), ctx);
				},
			},
			abort: async () => {
				throw new Error("held result must win");
			},
		});
		const { harness } = await openTasks(storage, [Parent, Middle, Leaf], noTimer);
		const parent = await start(harness, Parent);
		harness.resume();
		await eventually(() =>
			storage.commits.some((batch) =>
				batch.some((write) => write.type === "task" && write.value.kind === Leaf.definition.name),
			),
		);
		const leaf = await ready.promise;
		const root = await harness.root(context);
		const held = storage.holdCommits();
		parentFinishes.resolve();
		await held.entered;
		const stop = root.abort(context);
		held.release();
		await eventually(async () => (await harness.getTask(leaf, context))?.state.status === "sleeping");
		expect((await harness.getTask(parent, context))?.state).toEqual({
			status: "completing",
			outcome: completed("saved result").outcome,
		});
		await eventually(() => !harness.wake().active);
		expect(await settled(stop)).toBe(false);
		expect(await settled(harness.waitForIdle(context))).toBe(false);
		expect(order).toEqual([]);
		await harness.signal(leaf, Confirm, "leaf", "stopped", context);
		await stop;
		expect(order).toEqual(["leaf", "middle"]);
		expect((await harness.waitForTask(parent, context)).state.outcome).toEqual(completed("saved result").outcome);
		await harness.close(context);
	});

	it("withdraws inputs in an owned conversation that is cancelling and leaves background work live", async () => {
		const storage = new ControlledStorage();
		const ready = deferred<{ conversationId: ConversationId; background: TaskId }>();
		const Work = defineSignal({ name: "test.owned-work", version: 1, schema: str() });
		const Background = defineTask<null, { phase: "work" }, null>({
			name: "test.background-while-cancelling",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: (_task, runtime, ctx) =>
					runtime.commit(
						() => ({
							status: "sleeping",
							checkpoint: { phase: "work" },
							signals: [Work.key("done")],
						}),
						ctx,
					),
			},
			abort: (_task, runtime, ctx) => runtime.commit(() => abortedWith("background stopped"), ctx),
		});
		const Owner = defineTask<null, { phase: "spawn" }, null>({
			name: "test.owns-cancelling-conversation",
			version: 1,
			initial: () => ({ phase: "spawn" }),
			phases: {
				spawn: async (task, runtime, ctx) => {
					let created: { conversationId: ConversationId; background: TaskId } | undefined;
					await runtime.commit(async (tx) => {
						const ownership = { kind: "task", taskId: task.id } as const;
						const conversation = await tx.createConversation({ ownership });
						const background = await tx.createTask(Background, null, {
							conversationId: conversation.id,
							ownership: { kind: "conversation" },
							background: true,
						});
						created = { conversationId: conversation.id, background };
						return undefined;
					}, ctx);
					ready.resolve(created!);
					await aborted(runtime.signal);
				},
			},
			abort: (_task, runtime, ctx) => runtime.commit(() => abortedWith("owner stopped"), ctx),
		});
		const setup = chatSetup();
		addTask(setup.registry, Owner);
		addTask(setup.registry, Background);
		addTool(
			setup.registry,
			defineTool({
				name: "external",
				description: "External work",
				parameters: obj({}),
				execute: async () => ({ kind: "sleep", signals: [Work.key("done")] }),
				cancel: async (_args, api) => {
					if (api.wake === undefined) return { kind: "sleep", signals: [Confirm.key("done")] };
				},
			}),
		);
		setup.faux.setResponses([fauxAssistantMessage([fauxToolCall("external", {})], { stopReason: "toolUse" })]);
		const { harness, root } = await openChat(storage, setup);
		const owner = await start(harness, Owner);
		harness.resume();
		const { conversationId, background } = await ready.promise;
		const child = (await harness.conversation(conversationId, context))!;
		await child.configure({ model: { provider: "faux", modelId: "faux-1" } }, context);
		await child.submit({ type: "input", content: "start" }, context);
		let toolId: TaskId | undefined;
		await eventually(async () => {
			const slot = (await harness.snapshot(LiveDoc, child.id, context))?.tools?.[0];
			toolId = slot?.taskId;
			return toolId !== undefined && slot?.status === "sleeping";
		});
		const queued = await child.submit({ type: "input", content: "later", whenBusy: "followUp" }, context);
		expect((await queued.status(context)).status).toBe("queued");
		const stop = root.abort(context);
		await eventually(async () => {
			const state = (await harness.getTask(toolId!, context))?.state;
			return state?.status === "sleeping" && state.mode === "abort";
		});
		expect(await queued.wait(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
		const steer = await child.submit(
			{ type: "input", content: "steer while cancelling", whenBusy: "steer" },
			context,
		);
		const keep = await child.submit({ type: "input", content: "keep", onAbort: "keep" }, context);
		const repeat = root.abort(context);
		expect(await steer.wait(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
		expect((await keep.status(context)).status).toBe("queued");
		expect(await settled(repeat)).toBe(false);
		expect((await harness.getTask(background, context))?.abortRequested).toBe(false);
		await harness.signal(toolId!, Confirm, "done", "ack", context);
		await Promise.all([stop, repeat]);
		expect((await harness.getTask(owner, context))?.state.status).toBe("terminal");
		expect((await harness.getTask(background, context))?.state.status).toBe("sleeping");
		await harness.abortTask(background, context);
		expect((await harness.waitForTask(background, context)).state.outcome).toEqual(
			abortedWith("background stopped").outcome,
		);
		await harness.close(context);
	});
});

// #12090
describe("cancellation wait recovery", () => {
	it("preserves an abandoned task's cancellation sleep when Stop upgrades its mark", async () => {
		const storage = new ControlledStorage();
		const Task = defineTask<null, { phase: "work" }, null>({
			name: "test.abandoned-cancellation",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: async () => {
					throw new Error("abandoned work ran");
				},
			},
			abort: async (task, runtime, ctx) => {
				expect(task.abortReason).toBe("restart");
				await runtime.sleepUntil(200, task.state.checkpoint, ctx);
			},
		});
		let clock = 100;
		const options = { ...noTimer, now: () => clock };
		let { harness, registry } = await openTasks(storage, [], options);
		const extension = defineExtension({ name: "abandoned", tasks: [Task] });
		registry.install(extension);
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(Task, null, { ...OWNED_BY_CONVERSATION, abandonOnRestart: true }),
			context,
		);
		await harness.close(context);
		({ harness } = await openTasks(storage.reopen(), [], { ...options, registry }));
		harness.resume();
		await eventually(() => !harness.wake().active && harness.wake().at === 200);
		const parked = await harness.getTask(id, context);
		expect(parked).toMatchObject({
			abortRequested: true,
			abortReason: "restart",
			state: { status: "sleeping", mode: "abort" },
		});
		registry.uninstall(extension);
		expect(await harness.abortTask(id, context)).toBe("marked");
		const upgraded = await harness.getTask(id, context);
		expect(upgraded?.state).toEqual(parked?.state);
		expect(upgraded?.abortReason).toBeUndefined();
		expect(harness.wake()).toEqual({ active: false, at: 200 });
		clock = 200;
		expect((await harness.waitForTask(id, context)).state.outcome).toEqual({
			status: "orphaned",
			reason: "missing_task",
		});
		await harness.close(context);
	});

	it("orphans an abandoned task without a cancellation sleep when its definition is missing and Stop upgrades it", async () => {
		const storage = new ControlledStorage();
		const Task = defineTask<null, { phase: "work" }, null>({
			name: "test.missing-abandoned-cancellation",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: { work: async () => {} },
			abort: async () => {},
		});
		let { harness } = await openTasks(storage, []);
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(Task, null, { ...OWNED_BY_CONVERSATION, abandonOnRestart: true }),
			context,
		);
		await harness.close(context);
		({ harness } = await openTasks(storage.reopen(), []));
		harness.resume();
		await eventually(async () => (await harness.getTask(id, context))?.abortReason === "restart");
		expect(await harness.abortTask(id, context)).toBe("marked");
		expect((await harness.waitForTask(id, context)).state.outcome).toEqual({
			status: "orphaned",
			reason: "missing_task",
		});
		await harness.close(context);
	});

	it("reopens and upgrades the same sleeping task only at its wake boundary", async () => {
		const storage = new ControlledStorage();
		const cancellation = deferred();
		let clock = 100;
		let calls = 0;
		let migrations = 0;
		const Task = defineTask<null, { phase: "work" }, string>({
			name: "test.reopen-stop",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: { work: (_task, runtime, ctx) => runtime.sleepUntil(1_000, { phase: "work" }, ctx) },
			abort: async (task, runtime, ctx) => {
				calls++;
				expect(task.state.wake).toBeUndefined();
				await runtime.sleepUntil(200, { phase: "work" }, ctx);
				cancellation.resolve();
				throw new Error("sleep was already committed");
			},
		});
		const Updated = defineTask<null, { phase: "work" }, string>({
			...Task.definition,
			version: 2,
			migrate: () => {
				migrations++;
				return { input: null, checkpoint: { phase: "work" as const } };
			},
			abort: (task, runtime, ctx) => {
				calls++;
				expect(task.state.wake).toEqual({ reason: "time" });
				return runtime.commit(() => abortedWith("done"), ctx);
			},
		});
		const options = { ...noTimer, now: () => clock };
		let { harness, registry, infos } = await openTasks(storage, [Task], options);
		const id = await start(harness, Task);
		harness.resume();
		await eventually(() => harness.wake().at === 1_000);
		await harness.abortTask(id, context);
		await cancellation.promise;
		await eventually(() => !harness.wake().active);
		const wait = await harness.getTask(id, context);
		expect(wait?.state).toEqual({ status: "sleeping", mode: "abort", checkpoint: { phase: "work" }, until: 200 });
		expect(infos).toEqual([]);
		await harness.close(context);

		for (let reopen = 0; reopen < 2; reopen++) {
			({ harness, registry } = await openTasks(storage.reopen(), [], options));
			harness.resume();
			await harness.abortTask(id, context);
			registry.install(defineExtension({ name: "tasks", tasks: [Updated] }));
			harness.resume();
			await harness.abortTask(id, context);
			await flush();
			expect(await harness.getTask(id, context)).toEqual(wait);
			expect(harness.wake()).toEqual({ active: false, at: 200 });
			expect([calls, migrations, wait?.restarts]).toEqual([1, 0, undefined]);
			if (reopen === 0) await harness.close(context);
		}
		clock = 200;
		expect((await harness.waitForTask(id, context)).state.outcome).toEqual(abortedWith("done").outcome);
		expect([calls, migrations]).toEqual([2, 1]);
		await harness.close(context);
	});

	it.each(["request", "signal", "reservation"] as const)(
		"recovers a crash after %s without repeating normal work",
		async (cut) => {
			const storage = new ControlledStorage();
			const blocked = deferred();
			const requests: string[] = [];
			let normalRuns = 0;
			const wakeLog: unknown[] = [];
			const task = (crashing: boolean) =>
				defineTask<null, { phase: "work" } | { phase: "confirm"; key: string }, string>({
					name: "test.cancel-crash",
					version: 1,
					initial: () => ({ phase: "work" }),
					phases: {
						work: (_task, runtime) => {
							normalRuns++;
							return aborted(runtime.signal);
						},
						confirm: async () => {
							throw new Error("cancellation ran as business work");
						},
					},
					abort: async (task, runtime, ctx) => {
						const wake = task.state.wake;
						if (task.state.checkpoint.phase === "work") {
							const key = await runtime.memo("remote-request", `cancel:${task.id}`, ctx);
							requests.push(key);
							if (crashing && cut === "request") {
								const held = storage.holdCommits();
								void held.entered.then(() => blocked.resolve());
							}
							await runtime.commit(
								() => ({
									status: "sleeping",
									checkpoint: { phase: "confirm", key },
									signals: [Confirm.key(key)],
								}),
								ctx,
							);
							return;
						}
						wakeLog.push(wake);
						const key = task.state.checkpoint.key;
						expect(await runtime.readSignal(Confirm, key, ctx)).toEqual({ kind: "value", value: "ack" });
						if (crashing && cut === "reservation") {
							blocked.resolve();
							await new Promise(() => {});
						}
						await runtime.commit(() => abortedWith("ack"), ctx);
					},
				});
			const first = task(true);
			let { harness } = await openTasks(storage, [first], noTimer);
			const id = await start(harness, first);
			harness.resume();
			await eventually(() => normalRuns === 1);
			await harness.abortTask(id, context);
			const key = `cancel:${id}`;
			if (cut === "signal") {
				await eventually(() => !harness.wake().active);
				await harness.close(context);
				({ harness } = await openTasks(storage.reopen(), [first], noTimer));
				expect(await harness.commit((tx) => tx.signal(id, Confirm, key, "ack"), context)).toBe("delivered");
				const held = storage.holdCommits();
				harness.resume();
				await held.entered;
			} else {
				if (cut === "reservation") {
					await eventually(() => !harness.wake().active);
					expect(await harness.signal(id, Confirm, key, "ack", context)).toBe("delivered");
				}
				await blocked.promise;
			}
			storage.crash();
			({ harness } = await openTasks(storage, [task(false)], noTimer));
			const recovered = await harness.getTask(id, context);
			const memo = `pi.cancel-signal.${Confirm.name}/${key}`;
			const wake = { reason: "signal", signal: Confirm.name, key };
			if (cut === "request") {
				expect(recovered).toMatchObject({
					restarts: 1,
					memos: { "remote-request": key },
					state: { status: "pending", checkpoint: { phase: "work" } },
				});
				harness.resume();
				await eventually(() => !harness.wake().active);
				await harness.signal(id, Confirm, key, "ack", context);
			} else if (cut === "signal") {
				expect(recovered).toMatchObject({
					memos: { [memo]: { version: 1, value: "ack" } },
					state: { status: "sleeping", mode: "abort" },
				});
				expect(recovered?.memos?.[memo]).not.toHaveProperty("consumed");
			} else {
				expect(recovered).toMatchObject({
					restarts: 1,
					memos: { [memo]: { consumed: true } },
					state: { status: "pending", wake },
				});
			}
			expect((await harness.waitForTask(id, context)).state.outcome).toEqual(abortedWith("ack").outcome);
			expect(requests).toEqual(cut === "request" ? [key, key] : [key]);
			expect(wakeLog).toEqual(cut === "reservation" ? [wake, wake] : [wake]);
			expect(normalRuns).toBe(1);
			await harness.close(context);
		},
	);

	it.each(["no progress", "throw", "throw after completion"] as const)(
		"applies abort step precedence after %s",
		async (behavior) => {
			const Task = defineTask<null, { phase: "work" }, null>({
				name: "test.abort-step",
				version: 1,
				initial: () => ({ phase: "work" }),
				phases: { work: async () => {} },
				abort: async (_task, runtime, ctx) => {
					if (behavior === "no progress") return;
					if (behavior === "throw after completion") await runtime.commit(() => abortedWith("done"), ctx);
					throw new Error("abort error");
				},
			});
			const { harness, infos } = await openTasks(new ControlledStorage(), [Task]);
			const id = await start(harness, Task);
			await harness.abortTask(id, context);
			const receipt = await harness.waitForTask(id, context);
			if (behavior === "throw after completion") {
				expect(receipt.state.outcome).toEqual(abortedWith("done").outcome);
				expect(infos).toEqual([]);
			} else {
				expect(receipt.state.outcome).toEqual({
					status: "faulted",
					error: {
						message:
							behavior === "throw"
								? "abort error"
								: `Abort handler of task ${id} returned without a terminal outcome`,
					},
				});
			}
			await harness.close(context);
		},
	);

	it("counts repeated crashes during an abort invocation against the restart limit", async () => {
		const storage = new ControlledStorage();
		const calls: number[] = [];
		const Task = defineTask<null, { phase: "work" }, null>({
			name: "test.abort-crash-limit",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: async () => {
					throw new Error("normal work ran");
				},
			},
			abort: async (task) => {
				calls.push(task.restarts ?? 0);
				await new Promise(() => {});
			},
		});
		const options = { settings: { restartLimit: 2 } };
		let { harness } = await openTasks(storage, [Task], options);
		const id = await start(harness, Task);
		await harness.abortTask(id, context);
		for (let attempt = 1; attempt <= 2; attempt++) {
			harness.resume();
			await eventually(() => calls.length === attempt);
			storage.crash();
			({ harness } = await openTasks(storage, [Task], options));
		}
		expect((await harness.waitForTask(id, context)).state.outcome).toMatchObject({
			status: "faulted",
			error: { detail: { reason: "restart_limit" } },
		});
		expect(calls).toEqual([0, 1]);
		await harness.close(context);
	});
});
