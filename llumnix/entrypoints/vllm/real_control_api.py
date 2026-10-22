"""Loopback-only control endpoints on the existing serving process."""
from __future__ import annotations

import asyncio
from fastapi import Request
from fastapi.responses import JSONResponse


def install_real_control_routes(app, get_client, metric_states, flush):
    @app.post('/real_control/{method}')
    async def real_control(method: str, request: Request):
        if not request.client or request.client.host not in ('127.0.0.1','::1','localhost','testclient'):
            return JSONResponse({'error':'control requires loopback peer'},status_code=403)
        payload=await request.json()
        client=get_client()
        try:
            if method=='flush':
                cluster=await client.manager.real_control_flush.remote()
                local=await asyncio.to_thread(flush,5.)
                queue=client.request_output_queue
                pending=queue.qsize()+len(client.request_stream)+sum(not s['done'].is_set() for s in metric_states.values())
                local['pending_delivery']=pending
                return JSONResponse({'cluster':cluster,'api':local})
            if method=='release_episode':
                keys=[k for k in metric_states if k[:2]==(payload['run_id'],payload['episode_id'])]
                if any(not metric_states[k]['done'].is_set() for k in keys):
                    raise RuntimeError('episode handlers still active')
                for key in keys:
                    metric_states.pop(key)
                return JSONResponse({'status':'released','removed':len(keys)})
            result=await client.manager.real_control.remote(method,payload)
            return JSONResponse(result)
        except (RuntimeError,ValueError,KeyError) as exc:
            return JSONResponse({'error':str(exc)},status_code=409)
        except Exception as exc:
            return JSONResponse({'error':'uncertain control RPC: '+type(exc).__name__},status_code=503)
