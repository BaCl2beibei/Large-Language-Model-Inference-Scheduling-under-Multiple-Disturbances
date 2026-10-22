# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import AsyncGenerator
from contextlib import asynccontextmanager
import time
import asyncio
import json

from dataclasses import asdict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.sampling_params import SamplingParams

from llumnix.arg_utils import LlumnixArgumentParser, LaunchArgs
from llumnix.entrypoints.setup import setup_ray_cluster, setup_llumnix
from llumnix.entrypoints.vllm.arg_utils import add_cli_args, get_args, VLLMEngineArgs
from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
from llumnix.logging.logger import init_logger
from llumnix.utils import (
    random_uuid, BackendType, LaunchMode,
    asyncio_wait_for_ray_remote_call_with_timeout,
)
from llumnix.config import get_llumnix_config
from llumnix.entrypoints.utils import is_gpu_available, LlumnixTraceInfo
from llumnix.constants import SERVER_TIMEOUT_KEEP_ALIVE, LLUMNIX_TRACE_HEADER, LLUMNIX_TRACE_REQUEST
from llumnix.request_output import LlumnixRequestOuput

# Code file with __main__ should set the logger name to inherit the llumnix logger configuration.
logger = init_logger("llumnix.entrypoints.vllm.api_server")

llumnix_client: LlumnixClientVLLM = None
active_benchmark_requests: dict[str, dict[str, asyncio.Event]] = {}
finished_benchmark_requests: set[str] = set()
cancelled_benchmark_ids: set[str] = set()


# pylint: disable=unused-argument
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    try:
        yield
    finally:
        llumnix_client.cleanup()


app = FastAPI(lifespan=lifespan)


# pylint: disable=unused-argument
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    server_id = llumnix_client.server_info.server_id
    logger.exception("Server {} caught exception: {}".format(server_id, type(exc).__name__))
    llumnix_client.cleanup()

    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "error": str(exc)
        }
    )


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    # collect and return request latencys
    request_trace_param = {
        LLUMNIX_TRACE_REQUEST: request.headers.get(LLUMNIX_TRACE_HEADER, "False").lower() in ('true', '1')
    }

    # Use LlumnixClientVLLM's generate and abort api to replace with vLLM AsyncLLMEngine's generate and abort api.
    results_generator = await llumnix_client.generate(
        prompt, sampling_params, request_id, **request_trace_param
    )

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for llumnix_request_output in results_generator.generator():
            llumnix_request_output: LlumnixRequestOuput = llumnix_request_output
            request_output = llumnix_request_output.get_engine_output()
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            request_processing_context = llumnix_request_output.request_processing_context
            if request_processing_context.enable_trace:
                request_processing_context.add_trace_timeline("api_server_generate_timestamp_end")
                llumnix_trace_info = LlumnixTraceInfo(
                    latencys=request_processing_context.trace_timeline.to_latency_breakdown_dict(),
                    token_timestamps=request_processing_context.trace_timeline,
                )
                ret["llumnix_trace_info"] = asdict(llumnix_trace_info)
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    llumnix_trace_infos = []
    async for llumnix_request_output in results_generator.generator():
        llumnix_request_output: LlumnixRequestOuput = llumnix_request_output
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await llumnix_client.abort(request_id)
            return Response(status_code=499)
        request_output = llumnix_request_output.get_engine_output()
        final_output = request_output
        request_processing_context = llumnix_request_output.request_processing_context
        if request_processing_context.enable_trace:
            request_processing_context.add_trace_timeline("api_server_generate_timestamp_end")
            llumnix_trace_info = LlumnixTraceInfo(
                latencys=request_processing_context.trace_timeline.to_latency_breakdown_dict(),
                token_timestamps=request_processing_context.trace_timeline,
            )
            llumnix_trace_infos.append(asdict(llumnix_trace_info))

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    if llumnix_trace_infos:
        ret["llumnix_trace_info"] = llumnix_trace_infos
    return JSONResponse(ret)

@app.post("/generate_benchmark")
async def generate_benchmark(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # Add some benchmark-related codes comparing to the generate API.
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    _ = request_dict.pop("stream", False)
    request_id = str(request_dict.pop("request_id", "") or random_uuid())
    if request_id in cancelled_benchmark_ids:
        cancelled_benchmark_ids.discard(request_id)
        return Response(status_code=499)
    sampling_params = SamplingParams(**request_dict)
    request_state = {"ready": asyncio.Event(), "done": asyncio.Event(),
                     "task": asyncio.current_task()}
    active_benchmark_requests[request_id] = request_state

    start = time.perf_counter()

    # collect and return request latencys
    request_trace_param = {
        LLUMNIX_TRACE_REQUEST: request.headers.get(LLUMNIX_TRACE_HEADER, "False").lower() in ('true', '1')
    }

    try:
        results_generator = await llumnix_client.generate(prompt, sampling_params, request_id, **request_trace_param)
        request_state["ready"].set()

        # Non-streaming case
        final_output = None
        per_token_latency = []
        per_token_latency_breakdown_list = []
        async for llumnix_request_output in results_generator.generator():
            llumnix_request_output: LlumnixRequestOuput = llumnix_request_output
            request_output = llumnix_request_output.get_engine_output()
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await llumnix_client.abort(request_id)
                return Response(status_code=499)
            now = time.perf_counter()
            per_token_latency.append([now, (now - start)*1000])
            start = now
            final_output = request_output
            request_processing_context = llumnix_request_output.request_processing_context
            if request_processing_context.enable_trace:
                request_processing_context.add_trace_timeline('api_server_generate_timestamp_end', now)
                per_token_latency_breakdown_list.append(request_processing_context.trace_timeline.to_latency_breakdown_dict())
        # Explicit benchmark aborts can close the Llumnix stream without
        # producing a final RequestOutput. This is an expected cancelled
        # request, not a server error that should tear down the shared client.
        if final_output is None:
            return Response(status_code=499)

        if llumnix_client.log_requests:
            llumnix_client.num_finished_requests += 1
            logger.info("Entrypoints finished request {}".format(request_id))
            logger.info("Entrypoints num_finished_requests: {}".format(llumnix_client.num_finished_requests))

        generation = final_output.outputs[0].text
        num_output_tokens = len(final_output.outputs[0].token_ids)
        num_input_tokens = len(final_output.prompt_token_ids)
        expected_resp_len = request_dict['max_tokens']
        if not max(expected_resp_len, 1) == max(num_output_tokens, 1):
            logger.warning(
                "request_id={}, expected_resp_len={}, num_output_tokens={}, num_input_tokens={}".format(
                    request_id, expected_resp_len, num_output_tokens, num_input_tokens)
            )
        ret = {
            'request_id': request_id,
            'generated_text': generation,
            'num_output_tokens_cf': num_output_tokens,
            'per_token_latency': per_token_latency,
        }
        if per_token_latency_breakdown_list:
            ret['per_token_latency_breakdown_list'] = per_token_latency_breakdown_list
        return JSONResponse(ret)
    except asyncio.CancelledError:
        # /abort_benchmark cancels the HTTP handler after the backend abort
        # RPC is acknowledged, so the shared server remains healthy.
        return Response(status_code=499)
    finally:
        request_state["ready"].set()
        request_state["done"].set()
        active_benchmark_requests.pop(request_id, None)
        finished_benchmark_requests.add(request_id)
        if len(finished_benchmark_requests) > 50000:
            finished_benchmark_requests.clear()
        if len(cancelled_benchmark_ids) > 50000:
            cancelled_benchmark_ids.clear()


@app.post("/abort_benchmark")
async def abort_benchmark(request: Request) -> Response:
    """Batch-abort outstanding benchmark requests and wait for their handlers to exit."""
    request_ids = list(dict.fromkeys(str(value) for value in (await request.json()).get("request_ids", [])))
    active_states = {}
    results = []
    for request_id in request_ids:
        state = active_benchmark_requests.get(request_id)
        if state is None:
            if request_id in finished_benchmark_requests:
                results.append({"request_id": request_id, "status": "already_finished"})
            else:
                # Prevent a late request handler from dispatching after the barrier.
                cancelled_benchmark_ids.add(request_id)
                results.append({"request_id": request_id, "status": "not_found_cancelled_if_late"})
        elif state["done"].is_set():
            results.append({"request_id": request_id, "status": "already_finished"})
        else:
            active_states[request_id] = state

    async def wait_until_ready(state: dict) -> None:
        await asyncio.wait_for(state["ready"].wait(), timeout=120.0)

    readiness = await asyncio.gather(
        *(wait_until_ready(state) for state in active_states.values()), return_exceptions=True
    )
    ready_ids = {
        request_id for (request_id, _), result in zip(active_states.items(), readiness)
        if not isinstance(result, BaseException)
    }
    unresolved = {
        request_id for (request_id, _), result in zip(active_states.items(), readiness)
        if isinstance(result, BaseException)
    }

    # vLLM/Llumlet accepts an iterable of request IDs. Send one Ray RPC per
    # worker instead of one RPC per request, which otherwise times out under
    # high-QPS phases with a large backlog.
    instance_groups = {}
    request_instances = {}
    for request_id in ready_ids:
        instance_ids, instances = llumnix_client._get_instance_for_abort(request_id)
        request_instances[request_id] = set(instance_ids)
        for instance_id, instance in zip(instance_ids, instances):
            group = instance_groups.setdefault(instance_id, {"instance": instance, "request_ids": set()})
            group["request_ids"].add(request_id)

    async def abort_instance(instance_id: str, group: dict):
        try:
            await asyncio_wait_for_ray_remote_call_with_timeout(
                group["instance"].abort, tuple(group["request_ids"]), timeout=600.0
            )
            return instance_id, None
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to batch-abort requests on instance %s", instance_id)
            return instance_id, exc

    group_results = await asyncio.gather(
        *(abort_instance(instance_id, group) for instance_id, group in instance_groups.items())
    )
    failed_instances = {instance_id for instance_id, error in group_results if error is not None}
    successful_instances = set(instance_groups) - failed_instances

    for request_id, state in active_states.items():
        if request_id not in ready_ids:
            continue
        involved_instances = request_instances.get(request_id, set())
        if not involved_instances or involved_instances.issubset(successful_instances):
            # Finishing the AsyncStream closes the handler with HTTP 499 and
            # releases the tracked client state after the backend acknowledges.
            if request_id in llumnix_client.request_stream:
                llumnix_client._clear_client_request_states(request_id)
            else:
                route_task = state["task"]
                if not route_task.done() and route_task is not asyncio.current_task():
                    route_task.cancel()
        else:
            unresolved.add(request_id)

    unfinished = {
        request_id: state for request_id, state in active_states.items()
        if not state["done"].is_set()
    }
    if unfinished:
        done_waits = [asyncio.create_task(state["done"].wait()) for state in unfinished.values()]
        _, pending_waits = await asyncio.wait(done_waits, timeout=120.0)
        if pending_waits:
            for waiter in pending_waits:
                waiter.cancel()
        for request_id, state in unfinished.items():
            if not state["done"].is_set():
                unresolved.add(request_id)
                route_task = state["task"]
                if not route_task.done() and route_task is not asyncio.current_task():
                    route_task.cancel()

    for request_id in request_ids:
        if request_id in unresolved:
            results.append({"request_id": request_id, "status": "timed_out_or_abort_failed"})
        elif not any(row["request_id"] == request_id for row in results):
            results.append({"request_id": request_id, "status": "aborted_or_finished"})
    return JSONResponse({"requested": len(request_ids), "results": results,
                         "unresolved_request_ids": sorted(unresolved),
                         "failed_instances": sorted(failed_instances),
                         "active_request_count": len(active_benchmark_requests)},
                        status_code=500 if unresolved else 200)


@app.get("/is_ready")
async def is_ready() -> bool:
    return await llumnix_client.is_ready()


if __name__ == "__main__":
    parser: LlumnixArgumentParser = LlumnixArgumentParser()

    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)
    parser.add_argument("--server-log-level", type=str, choices=['debug', 'info', 'warning', 'error'])
    parser = add_cli_args(parser)
    cli_args = parser.parse_args()
    llumnix_config = get_llumnix_config(cli_args.config_file, args=cli_args)

    entrypoints_args, manager_args, instance_args, engine_args = get_args(llumnix_config, LaunchMode.LOCAL, parser, cli_args)
    backend_type = BackendType.VLLM if not instance_args.simulator_mode else BackendType.SIM_VLLM
    launch_args = LaunchArgs(launch_mode=LaunchMode.LOCAL, backend_type=backend_type)
    vllm_engine_args: VLLMEngineArgs = VLLMEngineArgs(engine_args, backend_type)

    # Launch or connect to the ray cluster for multi-node serving.
    setup_ray_cluster(entrypoints_args)

    # if gpu is not available, it means that this node is head pod without any llumnix components.
    if is_gpu_available():
        entrypoints_context = setup_llumnix(entrypoints_args, manager_args, instance_args, vllm_engine_args, launch_args)
        # Start the api server after all the components of llumnix are ready.
        loop = asyncio.new_event_loop()
        llumnix_client = LlumnixClientVLLM(entrypoints_context, loop)
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(app,
            host=entrypoints_args.host,
            port=entrypoints_args.port,
            log_level=entrypoints_args.server_log_level,
            timeout_keep_alive=SERVER_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=entrypoints_args.ssl_keyfile,
            ssl_certfile=entrypoints_args.ssl_certfile
        )
        server = uvicorn.Server(config)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
