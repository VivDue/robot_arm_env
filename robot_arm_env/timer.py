import time

class Timer:
  """
  A simple start-stop timer using time.time().
  """
  def __init__(self):
    self.start_time = None

  def start(self):
    """
    Starts the timer.
    """
    self.start_time = time.time()

  def stop(self):
    """
    Stops the timer and returns the elapsed time in seconds.
    """
    if self.start_time is None:
      raise RuntimeError("Timer is not started yet.")
    elapsed_time = time.time() - self.start_time
    self.start_time = None  # Reset start time for next usage
    return elapsed_time

# Example usage
timer = Timer()

timer.start()
# Your code to be timed here
time.sleep(4)  # Simulate some work

elapsed_time = timer.stop()
print(f"Elapsed time: {elapsed_time:.2f} seconds")