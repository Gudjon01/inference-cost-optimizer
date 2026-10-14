"""The replay lab: replays recorded traffic shape against candidate engine configs.

A run drains one replica, restarts its engine under each candidate config,
ramps the recorded workload until a latency limit breaks, and puts the
original back. The cheapest config that holds the limits wins.
"""
