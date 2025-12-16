from fastapi import APIRouter, HTTPException, Request, Body, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse
from app.api.models import (
    ClaudeResponse, StreamRequest, TaskResponse, TaskStatusResponse, 
    GeminiImageResponse, TrellisRequest, TrellisResponse, TrellisInput
)
from app.tasks.claude_tasks import ClaudePromptTask, ClaudeEditTask
from app.tasks.gemini_tasks import GeminiPromptTask, GeminiImageGenerationTask
from app.tasks.cerebras_tasks import get_cerebras_client
from app.core.redis import redis_service
from app.core.config import settings
import json
import uuid
import asyncio
from typing import Dict, Any
from celery.result import AsyncResult
import re
import httpx
import os
from fastapi import BackgroundTasks

# Create the router
router = APIRouter()

# Trellis API URL (deprecated - now using 302.ai)
# TRELLIS_API_URL = "https://api.piapi.ai/api/v1/task"

def convert_base64_to_data_url(base64_data: str, media_type: str = "image/png") -> str:
    """Convert base64 string to data URL format.
    
    Args:
        base64_data: Base64 string (may or may not include data URL prefix)
        media_type: Media type for the data URL (default: image/png)
        
    Returns:
        Data URL string in format: data:image/png;base64,{base64_data}
    """
    # #region agent log
    import time
    import json
    log_data = {
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "E",
        "location": "routes.py:convert_base64_to_data_url:entry",
        "message": "Converting base64 to data URL",
        "data": {
            "input_length": len(base64_data),
            "has_comma": "," in base64_data,
            "media_type": media_type
        },
        "timestamp": int(time.time() * 1000)
    }
    try:
        with open("d:\\ziliao\\1216\\vibe-draw\\.cursor\\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    
    # Remove data URL prefix if present
    if "," in base64_data:
        base64_data = base64_data.split(",")[-1]
    
    result = f"data:{media_type};base64,{base64_data}"
    
    # #region agent log
    log_data2 = {
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "E",
        "location": "routes.py:convert_base64_to_data_url:exit",
        "message": "Data URL conversion complete",
        "data": {
            "output_length": len(result),
            "output_prefix": result[:50]
        },
        "timestamp": int(time.time() * 1000)
    }
    try:
        with open("d:\\ziliao\\1216\\vibe-draw\\.cursor\\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data2, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    
    return result

async def call_302ai_trellis_api(
    image_url: str,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 12,
    slat_guidance_strength: int = 3,
    slat_sampling_steps: int = 12,
    mesh_simplify: float = 0.95,
    texture_size: int = 1024
) -> Dict[str, Any]:
    """Call 302.ai Trellis API for image-to-3D generation.
    
    Args:
        image_url: Image URL in data URL format
        ss_guidance_strength: SS guidance strength (default: 7.5)
        ss_sampling_steps: SS sampling steps (default: 12)
        slat_guidance_strength: SLAT guidance strength (default: 3)
        slat_sampling_steps: SLAT sampling steps (default: 12)
        mesh_simplify: Mesh simplify factor (default: 0.95)
        texture_size: Texture size (default: 1024)
        
    Returns:
        API response as dictionary with model_mesh and timings
    """
    # #region agent log
    import time
    log_data = {
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "A",
        "location": "routes.py:call_302ai_trellis_api:entry",
        "message": "Calling 302.ai Trellis API",
        "data": {
            "image_url_length": len(image_url),
            "image_url_prefix": image_url[:50] if len(image_url) > 50 else image_url,
            "api_base_url": settings.API_302AI_BASE_URL,
            "api_endpoint": f"{settings.API_302AI_BASE_URL}/302/submit/trellis",
            "has_api_key": bool(settings.TRELLIS_API_KEY),
            "api_key_length": len(settings.TRELLIS_API_KEY) if settings.TRELLIS_API_KEY else 0,
            "ss_guidance_strength": ss_guidance_strength,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_guidance_strength": slat_guidance_strength,
            "slat_sampling_steps": slat_sampling_steps,
            "mesh_simplify": mesh_simplify,
            "texture_size": texture_size
        },
        "timestamp": int(time.time() * 1000)
    }
    try:
        with open("d:\\ziliao\\1216\\vibe-draw\\.cursor\\debug.log", "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    
    if not settings.TRELLIS_API_KEY:
        raise ValueError("302.ai API key (TRELLIS_API_KEY) not configured")
    
    # #region agent log
    request_body = {
        "image_url": image_url,
        "ss_guidance_strength": ss_guidance_strength,
        "ss_sampling_steps": ss_sampling_steps,
        "slat_guidance_strength": slat_guidance_strength,
        "slat_sampling_steps": slat_sampling_steps,
        "mesh_simplify": mesh_simplify,
        "texture_size": texture_size
    }
    import json as json_module
    request_body_size = len(json_module.dumps(request_body))
    log_data2 = {
        "sessionId": "debug-session",
        "runId": "run1",
        "hypothesisId": "A",
        "location": "routes.py:call_302ai_trellis_api:before_request",
        "message": "Before HTTP request",
        "data": {
            "request_body_size_bytes": request_body_size,
            "request_body_size_mb": round(request_body_size / (1024 * 1024), 2),
            "timeout_seconds": 120.0
        },
        "timestamp": int(time.time() * 1000)
    }
    try:
        with open("d:\\ziliao\\1216\\vibe-draw\\.cursor\\debug.log", "a", encoding="utf-8") as f:
            f.write(json_module.dumps(log_data2, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            # #region agent log
            log_data3 = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B",
                "location": "routes.py:call_302ai_trellis_api:http_request_start",
                "message": "Starting HTTP POST request",
                "data": {
                    "url": f"{settings.API_302AI_BASE_URL}/302/submit/trellis",
                    "has_auth_header": True
                },
                "timestamp": int(time.time() * 1000)
            }
            try:
                with open("d:\\ziliao\\1216\\vibe-draw\\.cursor\\debug.log", "a", encoding="utf-8") as f:
                    f.write(json_module.dumps(log_data3, ensure_ascii=False) + "\n")
            except Exception:
                pass
            # #endregion
            
            response = await client.post(
                f"{settings.API_302AI_BASE_URL}/302/submit/trellis",
                headers={
                    "Authorization": f"Bearer {settings.TRELLIS_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=request_body
            )
            
            # #region agent log
            log_data4 = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "C",
                "location": "routes.py:call_302ai_trellis_api:http_response",
                "message": "HTTP response received",
                "data": {
                    "status_code": response.status_code,
                    "response_size": len(response.content) if hasattr(response, 'content') else 0
                },
                "timestamp": int(time.time() * 1000)
            }
            try:
                with open("d:\\ziliao\\1216\\vibe-draw\\.cursor\\debug.log", "a", encoding="utf-8") as f:
                    f.write(json_module.dumps(log_data4, ensure_ascii=False) + "\n")
            except Exception:
                pass
            # #endregion
            
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        # #region agent log
        log_data5 = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "D",
            "location": "routes.py:call_302ai_trellis_api:request_error",
            "message": "HTTP request error",
            "data": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_repr": repr(e)
            },
            "timestamp": int(time.time() * 1000)
        }
        try:
            with open("d:\\ziliao\\1216\\vibe-draw\\.cursor\\debug.log", "a", encoding="utf-8") as f:
                f.write(json_module.dumps(log_data5, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
        raise

async def get_task_result(task_id: str) -> Dict[str, Any]:
    """Get the result of a task from Redis or Celery."""
    # Try to get the result from Redis
    result_json = redis_service.get_value(f"task_response:{task_id}")
    
    if result_json:
        # Parse the result from Redis
        return json.loads(result_json)
    
    # Check if the task exists in Celery
    task_result = AsyncResult(task_id)
    
    if task_result.state == "PENDING":
        return {"status": "pending"}
    elif task_result.state == "FAILURE":
        return {
            "status": "error",
            "error": str(task_result.result),
            "error_type": type(task_result.result).__name__
        }
    elif task_result.state == "SUCCESS":
        # The task completed but the result wasn't in Redis
        return task_result.result
    else:
        return {"status": task_result.state.lower()}

@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of an asynchronous task."""
    result = await get_task_result(task_id)
    
    status = "completed" if result.get("status") not in ["pending", "failed"] else result.get("status")
    
    # Determine response type based on result content
    response_model = None
    if status == "completed":
        if "images" in result:
            # It's an image generation response
            response_model = GeminiImageResponse(**result)
        else:
            # It's a text generation response
            response_model = ClaudeResponse(**result)
    
    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        result=response_model
    )

@router.post("/queue/{type}", response_model=TaskResponse)
async def queue_task(type: str, request: StreamRequest):
    """Start a task based on the specified type.
    
    Types:
    - 3d: Uses Claude 3.7 for 3D generation
    - 3d_magic: For 3D magic generation (unimplemented)
    - image: For image generation using Gemini Imagen
    - extract_object: For object extraction (unimplemented)
    - llama: Uses Cerebras LLaMA model
    - edit: Uses Claude 3.7 to edit existing Three.js code
    """
    # Generate a task ID if not provided
    task_id = request.task_id or str(uuid.uuid4())
    
    # Handle different task types
    if type == "3d":
        # Use the existing Claude implementation
        ClaudePromptTask.apply_async(
            args=[
                task_id,
                request.image_base64,
                request.prompt,
                request.system_prompt,
                request.max_tokens,
                request.temperature,
                request.additional_params
            ],
            task_id=task_id
        )
    elif type == "edit":
        # Validate Three.js code is provided in additional_params
        if not request.threejs_code:
            raise HTTPException(status_code=400, detail="Three.js code is required for editing")
        
        # At least one of image or prompt must be provided
        if not request.image_base64 and not request.prompt:
            raise HTTPException(status_code=400, detail="At least one of image or text prompt must be provided")
        
        # Use the ClaudeEditTask for code editing
        ClaudeEditTask.apply_async(
            args=[
                task_id,
                request.threejs_code,
                request.image_base64,
                request.prompt,
                request.system_prompt,
                request.max_tokens,
                request.temperature,
                request.additional_params
            ],
            task_id=task_id
        )
    elif type == "3d_magic":
        # TODO: Implement 3D magic generation
        pass
    elif type == "image":
        # Check if we're generating images or processing an image with text
        if request.image_base64:
            # Image is required for GeminiImageGenerationTask
            GeminiImageGenerationTask.apply_async(
                args=[
                    task_id,
                    request.image_base64,
                    request.prompt,
                    request.system_prompt,
                    request.max_tokens,
                    request.temperature,
                    request.additional_params
                ],
                task_id=task_id
            )
        else:
            # Error - image is required
            raise HTTPException(status_code=400, detail="Image base64 is required for image generation")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task type: {type}")
    
    # Return the task ID for SSE subscription
    return TaskResponse(task_id=task_id)

async def event_generator(task_id: str, request: Request):
    """Generate SSE events from Redis pub/sub."""
    # Subscribe to the Redis channel
    pubsub = redis_service.subscribe(f"task_stream:{task_id}")
    
    try:
        # Check if the client is still connected
        while not await request.is_disconnected():
            # Get message from Redis pub/sub
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            
            if message:
                data = json.loads(message["data"])
                event_type = data.get("event")
                event_data = data.get("data")
                
                # Yield the event
                yield {
                    "event": event_type,
                    "data": json.dumps(event_data)
                }
                
                # If this is the completion event, exit the loop
                if event_type in ["complete", "error"]:
                    break
                    
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.01)
            
    except Exception as e:
        # Yield an error event
        yield {
            "event": "error",
            "data": json.dumps({
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "task_id": task_id
            })
        }
    finally:
        # Always unsubscribe from the channel
        pubsub.unsubscribe(f"task_stream:{task_id}")
        pubsub.close()

@router.get("/subscribe/{task_id}")
async def subscribe_claude_events(task_id: str, request: Request):
    """Stream events from a Claude 3.7 task."""
    # Return an event source response
    return EventSourceResponse(event_generator(task_id, request))

@router.post("/cerebras/parse")
async def parse_code_with_cerebras(code: str = Body(..., media_type="text/plain")):
    """Direct endpoint to parse code using Cerebras LLaMA model without SSE.
    
    Takes a plain text body containing the code to be parsed and returns the result directly.
    """
    # Initialize Cerebras client
    client = await get_cerebras_client()
    
    # Prepare the message parameters
    messages = [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": """You are provided with a JavaScript snippet containing a Three.js scene. Extract only the main 3D object creation code, including relevant geometries, materials, meshes, and groups. Completely remove all unrelated elements such as the scene, renderer, camera, lighting, ground planes, animation loops, event listeners, orbit controls, and window resize handling.

Present the resulting code directly, ending with a single statement explicitly returning only the main object (THREE.Mesh or THREE.Group) that was created.

Do not wrap the code in a function or module. Do not import anything.
""" + code
        }
    ]
    
    # Send the request to Cerebras
    response = await client.chat.completions.create(
        model="llama3.3-70b",
        messages=messages,
        max_tokens=4096,
        temperature=0.2,
        top_p=1
    )
    
    # Extract and clean the content
    raw_content = response.choices[0].message.content
    
    # Find code blocks marked with ```javascript ... ```
    code_blocks = re.findall(r'```(?:javascript)?(.*?)```', raw_content, re.DOTALL)
    
    # Use the first found code block, or fallback to the full content if no blocks found
    content = code_blocks[0].strip() if code_blocks else raw_content
    
    # Return the parsed code directly
    return {
        "status": "success",
        "content": content,
        "model": response.model,
        "usage": {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0)
        }
    }

async def process_302ai_trellis_task(task_id: str, image_url: str, trellis_input: TrellisInput):
    """Background task to process 302.ai Trellis API call and store result in Redis.
    
    Args:
        task_id: Task ID for tracking
        image_url: Image URL in data URL format
        trellis_input: Trellis input parameters from the request
    """
    try:
        # Publish start event
        redis_service.publish_start_event(task_id)
        
        # Map parameters from frontend format to 302.ai format
        # Frontend defaults: ss_sampling_steps=50, slat_sampling_steps=50
        # 302.ai defaults: ss_sampling_steps=12, slat_sampling_steps=12
        # Use frontend values if provided, otherwise use 302.ai defaults
        ss_sampling_steps = trellis_input.ss_sampling_steps if trellis_input.ss_sampling_steps else 12
        slat_sampling_steps = trellis_input.slat_sampling_steps if trellis_input.slat_sampling_steps else 12
        
        # Call 302.ai Trellis API
        response = await call_302ai_trellis_api(
            image_url=image_url,
            ss_guidance_strength=trellis_input.ss_guidance_strength or 7.5,
            ss_sampling_steps=ss_sampling_steps,
            slat_guidance_strength=trellis_input.slat_guidance_strength or 3,
            slat_sampling_steps=slat_sampling_steps,
            mesh_simplify=0.95,  # 302.ai default
            texture_size=1024    # 302.ai default
        )
        
        # Extract model mesh URL from response
        model_mesh_url = response.get("model_mesh", {}).get("url")
        
        if not model_mesh_url:
            raise ValueError("No model_mesh.url in 302.ai response")
        
        # Prepare success response in format expected by frontend
        result = {
            "status": "completed",
            "message": "Task completed successfully",
            "data": model_mesh_url,
            "full_response": response
        }
        
        # Publish completion event and store result
        redis_service.publish_complete_event(task_id, result)
        redis_service.store_response(task_id, result)
        
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors from 302.ai API
        error_detail = f"302.ai Trellis API error: {e.response.status_code}"
        try:
            error_json = e.response.json()
            if "error" in error_json:
                if isinstance(error_json["error"], dict) and "message" in error_json["error"]:
                    error_detail = error_json["error"]["message"]
                elif isinstance(error_json["error"], str):
                    error_detail = error_json["error"]
            elif "message" in error_json:
                error_detail = error_json["message"]
        except Exception:
            pass
        
        error_response = {
            "status": "error",
            "error": error_detail,
            "error_type": "HTTPStatusError",
            "task_id": task_id
        }
        
        try:
            redis_service.publish_error_event(task_id, Exception(error_detail))
            redis_service.store_response(task_id, error_response)
        except Exception:
            pass
        
    except httpx.RequestError as e:
        # Handle network errors
        error_response = {
            "status": "error",
            "error": f"Error connecting to 302.ai Trellis API: {str(e)}",
            "error_type": "RequestError",
            "task_id": task_id
        }
        
        try:
            redis_service.publish_error_event(task_id, e)
            redis_service.store_response(task_id, error_response)
        except Exception:
            pass
        
    except Exception as e:
        # Prepare error response
        error_response = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "task_id": task_id
        }
        
        try:
            redis_service.publish_error_event(task_id, e)
            redis_service.store_response(task_id, error_response)
        except Exception:
            pass  # Ignore Redis errors at this point

@router.post("/trellis/task", response_model=Dict[str, Any])
async def create_trellis_task(request_data: TrellisRequest, background_tasks: BackgroundTasks):
    """Create a task for 302.ai Trellis API image-to-3D generation.
    
    This endpoint accepts the same format as before for compatibility, but now uses 302.ai API.
    
    Args:
        request_data: The data containing image and parameters
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Dict[str, Any]: Task ID and status for WebSocket tracking
    """
    # Check if API key is available
    if not settings.TRELLIS_API_KEY:
        raise HTTPException(status_code=500, detail="302.ai API key (TRELLIS_API_KEY) not configured")
    
    # Generate a task ID
    task_id = str(uuid.uuid4())
    
    # Convert base64 image to data URL format
    base64_image = request_data.input.image
    image_url = convert_base64_to_data_url(base64_image)
    
    # Add background task to process the 302.ai API call
    background_tasks.add_task(
        process_302ai_trellis_task,
        task_id=task_id,
        image_url=image_url,
        trellis_input=request_data.input
    )
    
    # Return task ID in format compatible with frontend
    return {
        "code": 200,
        "data": {
            "task_id": task_id
        },
        "message": "Task submitted successfully"
    }

@router.websocket("/trellis/task/ws/{task_id}")
async def trellis_task_status_websocket(websocket: WebSocket, task_id: str):
    """WebSocket endpoint that monitors 302.ai Trellis task status from Redis.
    
    Args:
        websocket: The WebSocket connection
        task_id: The ID of the task to monitor
        
    Returns:
        Streams the task status and result via WebSocket
    """
    await websocket.accept()
    
    # Flag to control polling loop
    continue_polling = True
    max_polling_time = 180  # Maximum polling time in seconds (3 minutes)
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Start polling loop
        while continue_polling:
            try:
                # Check if we've exceeded max polling time
                elapsed_time = asyncio.get_event_loop().time() - start_time
                if elapsed_time > max_polling_time:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Task timed out",
                        "data": None
                    })
                    continue_polling = False
                    break
                
                # Try to get the result from Redis
                result = redis_service.get_response(task_id)
                
                if result:
                    # Task has completed or failed
                    status = result.get("status", "unknown")
                    
                    if status == "completed" or status == "success":
                        # Task completed successfully
                        await websocket.send_json({
                            "status": "completed",
                            "message": result.get("message", "Task completed successfully"),
                            "data": result.get("data"),  # This is the model_mesh.url
                            "full_response": result.get("full_response")
                        })
                        continue_polling = False
                    elif status == "error" or status == "failed":
                        # Task failed
                        error_message = result.get("error") or result.get("message", "Task processing failed")
                        await websocket.send_json({
                            "status": "failed",
                            "message": error_message,
                            "data": None,
                            "full_response": result
                        })
                        continue_polling = False
                    else:
                        # Still processing
                        await websocket.send_json({
                            "status": "processing",
                            "message": f"Task is {status}, waiting for completion",
                            "data": None
                        })
                else:
                    # No result yet, task is still processing
                    await websocket.send_json({
                        "status": "processing",
                        "message": "Task is being processed, please wait...",
                        "data": None
                    })
                
                # Check for messages from the client
                try:
                    # Wait for a message from the client with a timeout
                    client_message = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    
                    # If the client sent "close", stop polling
                    if client_message == "close":
                        continue_polling = False
                except asyncio.TimeoutError:
                    # No message from client, continue polling
                    pass
                
                # If we're still polling, wait 2 seconds before the next poll
                if continue_polling:
                    await asyncio.sleep(2)
                
            except Exception as e:
                # Handle errors during polling
                await websocket.send_json({
                    "status": "error",
                    "message": f"Error checking task status: {str(e)}",
                    "data": None
                })
                # Wait before retrying
                await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        # Client disconnected, exit the loop
        continue_polling = False
    except Exception as e:
        # Handle unexpected errors
        try:
            await websocket.send_json({
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "data": None
            })
        except Exception:
            # If we can't send the error, just exit
            pass
    finally:
        # Ensure the WebSocket is closed when we're done
        try:
            await websocket.close()
        except Exception:
            # If the connection is already closed, ignore the error
            pass
