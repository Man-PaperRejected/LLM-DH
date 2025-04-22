from multiprocessing import Process, Queue as MpQueue # Use multiprocessing Queue and Process
import time
import logging
import logging.handlers

def ds_worker(ask_q: MpQueue, answer_q: MpQueue, log_q: MpQueue):
    """Target function for the LLM simulator process."""
    queue_handler = logging.handlers.QueueHandler(log_q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # 移除默认handlers，防止重复
    root_logger.addHandler(queue_handler)
    logger = logging.getLogger("LLM")
    logger.info("LLM Simulator process started.")
    while True:
        try:
            ask_text = ask_q.get() # Blocking get
            if ask_text is None: # Sentinel value to stop the process
                 logger.info("LLM Simulator process received stop signal. Exiting.")
                 break
            logger.info(f"LLM Simulator: Received text: '{ask_text}'")

            # Simulate processing time using config values (use time.sleep)
            simulated_delay = 2 # Example delay
            time.sleep(simulated_delay) # Blocking sleep

            answer_text = f"user: {ask_text}" # Simple prefix simulation
            logger.info(f"LLM Simulator: Generated answer: '{answer_text}'")

            answer_q.put(answer_text) # Blocking put
            # ask_q.task_done() # task_done/join not typically used across processes like this

        except EOFError: # Can happen if the queue is closed unexpectedly
             logger.warning("LLM Simulator process encountered EOFError on queue. Exiting.")
             break
        except Exception as e:
            # Log error but continue running if possible
            logger.error(f"Error in LLM Simulator process: {e}", exc_info=True)
            time.sleep(1) # Avoid busy-looping on errors

