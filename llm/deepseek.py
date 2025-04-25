from multiprocessing import Process, Queue as MpQueue # Use multiprocessing Queue and Process
import time
import logging
import logging.handlers
import os
import openai
from dotenv import load_dotenv
def ds_worker(ask_q: MpQueue, answer_q: MpQueue, llm_queue: MpQueue, log_q: MpQueue):
    """Target function for the LLM simulator process."""
    queue_handler = logging.handlers.QueueHandler(log_q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # 移除默认handlers，防止重复
    root_logger.addHandler(queue_handler)
    logger = logging.getLogger("LLM")
    logger.info("LLM process started.")
    try:
        load_dotenv()
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE") 
        MODEL_NAME = "deepseek-chat" 
        client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
        )
        logger.info(f"connect to DeepSeek API endpoint: {DEEPSEEK_API_BASE}")
        logger.info(f"use model: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"LLM init failed {e}")
        
    conversation_history = [
        {"role": "system", "content": "你是一个乐于助人的AI助手。回答不要带表情等其他符号"} 
    ]
    while True:
        try:
            ask_text = ask_q.get() # Blocking get
            if ask_text is None: # Sentinel value to stop the process
                 logger.info("LLM Simulator process received stop signal. Exiting.")
                 break
            logger.info(f"LLM: Received text: '{ask_text}'")
            conversation_history.append({"role": "user", "content": ask_text})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=conversation_history,
                    max_tokens=1024, # 可以调整生成的最大长度
                    temperature=0.7, # 控制创造性，0 表示更确定性，1 表示更随机
                    stream=False,     # 设置为 False 以获取完整响应
                )

                assistant_reply = response.choices[0].message.content
 
                conversation_history.append({"role": "assistant", "content": assistant_reply})

            except openai.APIConnectionError as e:
                logger.error(f"API connection error: {e}")
            except openai.RateLimitError as e:
                logger.error(f"Rate limition: {e}")
            except openai.APIStatusError as e:
                logger.error(f"API State Error (HTTP {e.status_code}): {e.response}")
            except Exception as e:
                logger.error(f"Unknown Error: {e}")
            
            answer_text = assistant_reply # Simple prefix simulation
            logger.info(f"LLM Simulator: Generated answer: '{answer_text}'")

            answer_q.put(answer_text) # Blocking put
            llm_queue.put(answer_text)
            # ask_q.task_done() # task_done/join not typically used across processes like this

        except EOFError: # Can happen if the queue is closed unexpectedly
             logger.warning("LLM Simulator process encountered EOFError on queue. Exiting.")
             break
        except Exception as e:
            # Log error but continue running if possible
            logger.error(f"Error in LLM Simulator process: {e}", exc_info=True)
            time.sleep(1) # Avoid busy-looping on errors

