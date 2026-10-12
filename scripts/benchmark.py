from cs336_basics.model import BasicsTransformerLM
import torch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import json
import sys
def benchmark(mode='full', device='cuda', batchsize=128, num_trials=5, **kwargs):

  warmup = 5

  data = torch.randint(0, kwargs['vocab_size'], (batchsize, kwargs['context_length']), device=device)
  target = torch.randint(0, kwargs['vocab_size'], (batchsize, kwargs['context_length']), device=device)

  model = BasicsTransformerLM(**kwargs)
  model.to(device)
  optimizer = AdamW(model.parameters())

  if mode == 'forward':
    with torch.no_grad():
      times = []
      for _ in range(warmup):
        model.forward(data)
      torch.cuda.synchronize()
      for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        model.forward(data)
        end_event.record()

        torch.cuda.synchronize()
        times.append((start_event.elapsed_time(end_event)))

      print(times)

  elif mode == 'forward_and_backward':
    times = []
    for _ in range(warmup):
      optimizer.zero_grad()
      pred = model.forward(data)
      loss = cross_entropy(pred, target)
      loss.backward()
    torch.cuda.synchronize()
    for trial in range(num_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
      optimizer.zero_grad()
      pred = model.forward(data)
      loss = cross_entropy(pred, target)
      loss.backward()
      end_event.record()

      torch.cuda.synchronize()
      times.append((start_event.elapsed_time(end_event)))
    print(times)

  elif mode == 'full':
    times = []
    for _ in range(warmup):
      optimizer.zero_grad()
      pred = model.forward(data)
      loss = cross_entropy(pred, target)
      loss.backward()
      optimizer.step()
    torch.cuda.synchronize()
    for trial in range(num_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
      optimizer.zero_grad()
      pred = model.forward(data)
      loss = cross_entropy(pred, target)
      loss.backward()
      optimizer.step()
      end_event.record()

      torch.cuda.synchronize()
      times.append((start_event.elapsed_time(end_event)))
    print(times)


if __name__ == "__main__":
  if len(sys.argv) < 3:
    sys.exit(1)

  config_path = sys.argv[1]
  mode = sys.argv[2]
  with open(config_path, "r") as c:
    config = json.load(c)

  benchmark(mode, **config)