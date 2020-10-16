---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Building a bot to trade foreign currency using reinforcement learning (Part 2)"
subtitle: "We'll create a OpenAI Gym environment to train a currency trading bot"
summary: ""
authors: []
tags: []
categories: []
date: 2019-08-13T16:31:50-04:00
lastmod: 2019-08-13T16:31:50-04:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
In [Part1]({{< ref "/post/ppo2-currency-trader-part2.md" >}}), we configured OANDA v20 and downloaded data to train our agent. Today, we are going to build a custom OpenAI Gym environment. 

Check out my full codes on [GitHub](https://github.com/pipatth/PPO2-currency-trader).

Gym is an open-sourced tool to train your reinforcement learning agent. Think of Gym as a training ground. The agent send an action to the environment. The environment processes it and reply with the next observation and reward. This cycle goes on until the training session (episode) ends. 

![RL cycle](https://gym.openai.com/assets/docs/aeloop-138c89d44114492fd02822303e6b4b07213010bb14ca5856d2d49d6b62d88e53.svg)

source: [https://gym.openai.com/docs/](https://gym.openai.com/docs/)

Gym provides many pre-built environments from the basic ones such as `CartPole-v0` to Atari old video games. Although Gym doesn't provide a pre-built environment for currency trading, we can get one by creating a subclass of `gym.Env` class.

But first we need to install Gym:
```
$ pip install gym
```

Before we start building an environment, let's understand how `gym.Env` class works. Adam King did a very good job explaining each method [here](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e). My code is based on his. More details about Gym can be found on Gym documentation [here](https://gym.openai.com/docs/)

Basically, there are two public methods that you need to write:

- `reset()`. This gets called when you want to start a new session.
- `step(action)`. You call this one when your agent takes an action.

Note: There is also a `render()` method but we skip it for now because we don't need to show anything on the screen. 

A simple run of `CartPole-v0` looks like this:
```
import gym
env = gym.make('CartPole-v0')
env.reset() # reset environment to start a session
for _ in range(1000): # run for 1,000 steps
    env.step(1) # take action 1
env.close()
```
Easy, right? Let's put a currency trading touch to it.

**`__init__()`**

We're going to initialize the environment with the data that we downloaded from [Part1]({{< ref "/post/ppo2-currency-trader-part2.md" >}}). We also need to add common parameters such as commission (good if we want to use the code in stock trading), slippage, and the maximum number of steps that a session is going to run. 

Currency trading is a little different from stock or Bitcoin. You have a regulatory margin requirement. OANDA published margin percentage and a guide how it calculates margin closeout [here](https://oanda.secure.force.com/AnswersSupport?urlName=How-to-Calculate-a-Margin-Closeout-1436196462931&language=en_US#one). In our case, I set margin requirement to 100% (i.e. you need to have equity to cover 100% of your position)

An `action_space` has two elements. One has three choices whether we `long`, `short`, `close all positions`, or `hold`. The other is the proportion that we long or short (based on total margin available). This can range from 1/10, 2/10, to 10/10. 

To make it easy to report portfolio performance, I define the size of `account_history` dataframe to 5 columns. We'll send out net asset value `nav`, units long/short, and proceeds in a pandas dataframe later.

```
# init
def __init__(
    self,
    df,
    commission=0.0,
    margin_req=1.00,
    closeout_req=0.5,
    slippage=0.0,
    initial=100000,
    serial=False,
    max_steps=200,
):
    super(TradeEnv, self).__init__()
    self.lookback_sz = 24
    self.commission = commission
    self.margin_req = margin_req
    self.closeout_req = closeout_req
    self.slippage = slippage
    self.initial = initial
    self.serial = serial
    self.max_steps = max_steps
    self.colTime = "time"
    self.cols = {
        "volume": "volume",
        "ask": "ask.c",
        "bid": "bid.c",
        "open": "mid.o",
        "high": "mid.h",
        "low": "mid.l",
        "close": "mid.c",
    }
    self.ac_cols = ["nav", "unit_long", "value_long", "unit_short", "value_short"]
    self.df = df.dropna().sort_values(self.colTime)
    self.action_space = spaces.MultiDiscrete([4, 10])
    self.df_sz = self.df.shape[1] - 1
    self.observation_space = spaces.Box(
        low=0,
        high=1,
        shape=(self.df_sz + len(self.ac_cols), self.lookback_sz + 1),
        dtype=np.float16,
    )
```
**`reset()`**

`reset()` job is to set `balance` to the seed money that we give the agent. The `units_open` and `prices_open` lists are queues to keep track of positions that we have. OANDA requires a FIFO (first-in, first-out) so for example, if you are closing half of your current position, OANDA will close the oldest one first.

We call `_reset_pos()` private method (details [here](https://github.com/pipatth/PPO2-currency-trader/blob/master/env.py)) to find the starting position of the dataframe. Remember that we have 30,000 training data points in Part1? This method randomly pick one starting point from that.

As I mentioned earlier, we create three pandas dataframes to help us track and report trades, prices, and the account. 

At the end of the method, it's a convention in Gym to get the next observation and pass it out. We call a `_next_obs()` method.

```
# reset
def reset(self):
    self.balance = self.initial
    self.units_open = []  # FIFO queue
    self.prices_open = []
    self._reset_pos()

    # account history df
    ac_array = np.repeat([[self.initial, 0, 0, 0, 0]], self.lookback_sz + 1, axis=0)
    self.ac_hist = pd.DataFrame(ac_array, columns=self.ac_cols)

    # price history df
    pr_cols = [self.cols[k] for k in self.cols.keys()]
    self.pr_hist = self.df.loc[
        self.start_pos - self.lookback_sz : self.start_pos, pr_cols
    ]
    self.pr_hist.columns = [k for k in self.cols.keys()]

    # trade history df
    tr_cols = ["step", "type", "unit", "total"]
    self.tr_hist = pd.DataFrame(None, columns=tr_cols)

    return self._next_obs()
```

**`_next_obs()`**

`_next_obs()` get observation (aka information) and pass it to the agent. Our agent will use this information as an input and come up with the next action. We can get creative with technical indicators (see https://www.ta-lib.org/) but let's keep things simple for now with OHLCV ("open", "high", "low", "close", "volume") and account history.

Similar to other neural networks, most algorithms run better when we scale the data. 

```
# get next obs
def _next_obs(self):
    # OHLCV info
    end = self.current_step + self.lookback_sz + 1
    pr_hist = (
        self.active_df.iloc[self.current_step : end]
        .drop(self.colTime, axis=1)
        .values
    )
    obs = preprocessing.StandardScaler().fit_transform(pr_hist).T

    # append scaled history
    scaled_hist = preprocessing.MinMaxScaler().fit_transform(self.ac_hist.T)
    obs = np.append(obs, scaled_hist[:, -(self.lookback_sz + 1) :], axis=0)
    return obs
```

**`step(action)`**

This method is called every time the agent is taking an action. After taking an action (`_take_action` method), we move `current_step` forward and inform the agent the reward and the next observation. One thing to note here is we force sell if episode ends.
```
# take a step forward
def step(self, action):
    nav_beg = self.get_nav()
    pr_row = self._get_pr_row()
    self._take_action(action, pr_row)
    self.remaining_steps -= 1
    self.current_step += 1
    # end of episode, force close
    if self.remaining_steps == 0:
        unit_to_close = -sum(self.units_open)  # reverse to close
        if unit_to_close > 0:
            price = pr_row["ask"]
            self._close_pos(unit_to_close, price)
        elif unit_to_close < 0:
            price = pr_row["bid"]
            self._close_pos(unit_to_close, price)
        self._reset_pos()
    obs = self._next_obs()
    reward = self.get_nav() - nav_beg
    done = self._is_closeout()

    # append next price to pr_hist
    pr_cols = [self.cols[k] for k in self.cols.keys()]
    pr_next = self.df.loc[self.start_pos + self.current_step, pr_cols]
    pr_next.index = [k for k in self.cols.keys()]
    self.pr_hist = self.pr_hist.append(pd.DataFrame(pr_next).T)

    return obs, reward, done, {}
```

**`_take_action(action, pr_row)`**

Instead of just long/short/close/hold, we have to be careful of the imposed margin requirement. `avail` indicates how much margin we have available after subtracting margin used by the positions that we already opened. 

Note: I didn't store margin used and NAV but wrote few private methods to compute them from positions opened.

If an action is to close all positions (`action_type == 2`), we don't need to care about margin, just reverse the number of units that we have and close them using the right bid/ask. Notice that we apply slippage here. 

If an action is either `0` or `1` which mean `long` or `short`, we check how many units we can take based on available margin. We need to round up the units to whole numbers here.

```
# action
def _take_action(self, action, pr_row):
    action_type = action[0]
    proportion = action[1] / 10
    avail = self.get_nav() - self._get_margin_used()  # margin available to use
    unit_filled = 0
    price = 0
    # close
    if action_type == 2:
        unit = -sum(self.units_open)
        if unit > 0:
            price = pr_row["ask"] * (1 + self.slippage)
        else:
            price = pr_row["bid"] * (1 - self.slippage)
        unit_filled = self._close_pos(unit, price)
    # take pos
    elif avail > 0:
        # long
        if action_type == 0:
            price = pr_row["ask"] * (1 + self.slippage)
            unit = int(avail / price * proportion)  # unit to long
            unit_filled = self._open_pos(unit, price)
        # short
        elif action_type == 1:
            price = pr_row["bid"] * (1 - self.slippage)
            unit = -int(avail / price * proportion)  # unit to short
            unit_filled = self._open_pos(unit, price)

    # record trade history
    if unit_filled != 0:
        tr = pd.DataFrame(
            [
                [
                    self.start_pos + self.current_step,
                    "long" if unit_filled > 0 else "short",
                    unit_filled,
                    unit_filled * price,
                ]
            ],
            columns=self.tr_hist.columns,
        )
        self.tr_hist = self.tr_hist.append(tr, ignore_index=True)

    # record account history
    if unit_filled > 0:
        unit_long = abs(unit_filled)
        unit_short = 0
    elif unit_filled < 0:
        unit_long = 0
        unit_short = abs(unit_filled)
    else:
        unit_long = 0
        unit_short = 0
    df_ = pd.DataFrame(
        [
            [
                self.get_nav(),
                unit_long,
                unit_long * price,
                unit_short,
                unit_short * price,
            ]
        ],
        columns=self.ac_hist.columns,
    )
    self.ac_hist = pd.concat([self.ac_hist, df_], ignore_index=True)
```

**`get_summary()`**

Lastly, we add another public method to get a summary of the account and the prices in pandas dataframe. This also make it easy to update metrics such as Sharpe or maximum drawdown later as the agent trades. 

```
# get summary
def get_summary(self):
    accounts = self.ac_hist.reset_index(drop=True)
    prices = self.pr_hist.reset_index(drop=True)
    accounts["gl"] = accounts["nav"].diff().fillna(0)
    accounts["ret"] = (accounts["gl"] / accounts["nav"].shift(1)).fillna(0)
    return pd.concat([prices, accounts], axis=1)
```

Voila! we have a trading environment. We can start training our bot in Part3. 
