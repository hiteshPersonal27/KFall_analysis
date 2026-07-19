# Ensemble Trigger Plan: Methodology and Build Rationale for the Voting Ensemble

## How to use this document

This document is both the plan (what to build) and the rationale (why each part works and
what each method is actually doing), so the implementation is done with understanding
rather than mechanically.

The task is a binary pre-impact fall trigger. At every frame the system decides fire or do not fire, using only past frames. Three detectors each make this decision in their own way, and their decisions are pooled by voting into one ensemble decision. The goal is to fire after a fall begins but before the body hits the ground, so a protective device could act in time.

Every mathematical term is defined where it appears.

---

## Background: what the signal looks like during a fall

A fall recorded at the lower back has a recognisable structure in the accelerometer magnitude ACC_M, which sits near 1.0 g during calm activity.

* In a forward fall the person can brace, so ACC_M shows a bump, a rise above baseline, partway through the fall.
* In a backward or lateral fall the person cannot brace, so ACC_M shows a dip, a fall toward zero g as the body descends without support.
* In all fall types ACC_M ends in a sharp impact spike when the body meets the ground.

The detection problem is to catch this developing pattern early, before the impact spike, and to not confuse it with ordinary activities that also cause acceleration changes, such as a stumble, a quick sit, or a jump. These risky activities are the main source of false alarms.

Two derived signals help. Fitting a small curve to a short sliding window of the signal gives the local slope, written b1, and the local curvature, written b2. The slope says whether the signal is rising or falling and how steeply. The curvature says how sharply the signal is bending. Both are near zero during calm activity and depart during a fall, which makes them cleaner to work with than the raw value.

---

## Detector 1: Threshold

### What it does

It fires the instant a chosen signal crosses a fixed cutoff. It has no memory. It judges only the current frame.

### The rule

```
fire when signal(t) crosses the cutoff c
```

Terms:

* `signal(t)` is the value of the chosen signal at the current frame t, for example ACC_M or VV.
* `c` is a fixed cutoff constant, chosen during tuning.

For a signal that rises during a fall, fire when it goes above c. For a signal that drops during a fall, such as the ACC_M dip, fire when it goes below c.

### Strengths and weaknesses

It is the simplest and fastest method and is fully explainable, since you can point at the exact value that crossed the line. Its weakness is that a single frame value carries no context. It misses slow falls whose values never cross a strong cutoff, and it false alarms on brief sharp movements such as a stumble that momentarily crosses the cutoff. In the ensemble it contributes speed and simplicity, and it acts as the baseline that the other methods must beat.

---

## Detector 2: CUSUM

### What it does

It fires when evidence of a sustained departure from baseline has accumulated past a limit. Unlike the threshold, it has memory. It reacts to a persistent shift built up over many frames, not to a single value.

### The rule

Maintain two running totals, one watching for upward drift and one for downward drift.

```
S_plus(t)  = max( 0 , S_plus(t-1)  + ( x(t) - mu ) - k )
S_minus(t) = max( 0 , S_minus(t-1) - ( x(t) - mu ) - k )

fire when S_plus(t) > h  or  S_minus(t) > h
```

Terms:

* `x(t)` is the chosen signal value at frame t.
* `mu` is the baseline mean, the normal resting value measured from the calm early part of the trial.
* `( x(t) - mu )` is the deviation, how far the current value sits from normal.
* `k` is the slack, a small constant subtracted so that ordinary noise is ignored and only deviation larger than k accumulates.
* `S_plus(t)` and `S_minus(t)` are the running totals of upward and downward deviation evidence.
* `S_plus(t-1)` and `S_minus(t-1)` are those same totals from the previous frame, which is what makes the totals accumulate over time.
* `max( 0 , ... )` sets a floor at zero, resetting a total whenever it would go negative, so a total only climbs under sustained one directional deviation.
* `h` is the decision limit a total must reach in order to fire, chosen during tuning.

### Why it differs from a threshold

Both use a cutoff, but the threshold cuts on the raw value at one instant, while CUSUM cuts on an accumulated total with memory. This is why CUSUM catches slow drifts that a value threshold misses, and resists single spike false alarms, since one spike does not accumulate before the total resets.

### Useful bonus

If S_plus fires the signal was rising, which matches a forward bump. If S_minus fires the signal was dropping, which matches a backward or lateral dip. Recording which total fired gives an indication of the fall direction family.

### Strengths and weaknesses

Expected to be stronger than the threshold on slow low acceleration falls and more robust to brief false triggers. Its weakness is that it assumes a reasonably steady baseline, so for falls during walking the baseline oscillates. Running it on the slope b1 rather than the raw signal reduces this problem, because the slope flattens routine gait oscillation.

---

## Detector 3: Shapelet

### What it does

It learns the short characteristic shape that best distinguishes a fall from normal activity, then fires when the incoming signal matches that shape closely enough. The learned shape is a visible curve, which makes this the most directly interpretable detector.

### Learning the shape, done once offline

From the training subjects, many short snippets are taken from fall trials and from normal trials. Each candidate snippet is scored by how well its distance separates falls from normal activity. A good snippet is close to fall signals, because the shape is present in falls, and far from normal signals, because the shape is absent from normal activity. The best snippet is kept as the shapelet. Optionally one shapelet is kept per fall direction family, a bump shapelet and a dip shapelet.

### The matching rule, per frame

```
d(t) = dist( shapelet , window ending at t )

fire when d(t) < c_match
```

Terms:

* `window ending at t` is the most recent L frames of the signal, where L is the shapelet length.
* `dist( shapelet , window )` is the sum over the L positions of the squared difference between the shapelet value and the window value at each position. A small value means the two curves have nearly the same shape.
* `d(t)` is that distance at the current frame, meaning how far the current window is from the learned fall shape.
* `c_match` is the match cutoff. When the distance falls below it, the fall shape is considered present, chosen during tuning.

### Strengths and weaknesses

Expected to give the best specificity, since it fires only on a specific learned shape rather than on a raw level, and it names the fall type through which shapelet matched. Its weakness is that the offline discovery step is computationally heavy, so an accelerated approach such as random sampling of candidate snippets is used. The runtime matching step itself is cheap.

---

## The ensemble: combining the three by voting

### The idea

All three detectors run at the same time on the same trial. At every frame each one outputs a vote, 1 for fire and 0 for do not fire. The votes are pooled into a single ensemble decision. This is a bagging style ensemble, meaning the members vote in parallel and the votes are pooled with no member depending on another.

### Vote pooling

```
V(t) = v_threshold(t) + v_cusum(t) + v_shapelet(t)
```

Term. `V(t)` is the number of detectors voting to fire at frame t, an integer from 0 to 3.

### The firing rules to compare

```
ANY        ensemble fires when V(t) is at least 1
MAJORITY   ensemble fires when V(t) is at least 2
UNANIMOUS  ensemble fires when V(t) is at least 3
```

The default is MAJORITY, the natural pooled vote. The first frame that satisfies the rule is the ensemble detection frame for that trial.

### Why voting can help, and the condition under which it does

Voting improves on a single method only when the methods make different mistakes. A false alarm from one detector is cancelled when the other two do not agree at that frame. This requires the detectors to have complementary errors. If instead all three tend to false alarm on the same activities, for example all three on a stumble, then voting cannot cancel that error, because the required disagreement does not occur.

This is why the build studies each detector alone first and produces a per activity false alarm breakdown. That breakdown reveals whether the errors are complementary. If the three false alarm on different activities, the voting ensemble is expected to raise specificity. If they false alarm on the same activities, voting will help little, and the members should instead be given different signals to force independence. Either outcome is informative.

---

## What success looks like

Success is measured by three numbers. Sensitivity is the share of falls caught before impact. Specificity is the share of normal activities correctly left alone. Lead time is the average warning time before impact, in milliseconds, where one frame is ten milliseconds at 100 Hz. A detection counts only if it happens before the impact frame.

The realistic expectation is that the ensemble raises specificity over the best single method by cancelling the false alarms where members disagree, while keeping a lead time close to the fastest member. The size of that gain depends on how independent the members are, which the false alarm breakdown will show directly.
