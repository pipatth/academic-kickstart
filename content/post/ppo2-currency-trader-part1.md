---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Building a bot to trade foreign currency using reinforcement learning (Part 1)"
subtitle: "We'll train an RL bot using OpenAI Gym to trade foreign currency and see how it performs on Dash app"
summary: ""
authors: []
tags: []
categories: []
date: 2019-08-12T15:28:41-04:00
lastmod: 2019-08-12T15:28:41-04:00
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
Today we're going to build an environment to train a reinforcement learning agent to trade foreign currency. We'll be using OpenAI Gym as our tool. Gym custom environment is very flexible to set up and you can apply the idea into other reinforcement learning projects.

So here's the list of what we're going to do:

- Set up OANDA v20 to download data from OANDA (Part1)
- Write an OpenAI Gym to simulate the foreign exchange market (Part2)
- Train an agent to trade using [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) algorithm (Part3)
- See how the agent performs on [Dash](https://dash.plot.ly/) app (Part3)

You can clone my code on [GitHub](https://github.com/pipatth/PPO2-currency-trader).

So here we go.

Let's begin by setting up an OANDA API so we can download historical data from OANDA. You can go to OANDA and sign up for a free FXTrade Practice account [here](https://www1.oanda.com/register/#/sign-up/demo). After you're done, OANDA will give you token information. You'll need to put the information in your .v20.conf file in your `$HOME` directory. 

Something like this:
{{< gist pipatth 2c04cf6e393388651cb4ac38dddd8000 ".v20.conf" >}}

OANDA provides a python library called [v20](https://github.com/oanda/v20-python). With this library, we don't have to write our own HTTP calls. Let's clone and install that on your virualenv environment. We also gonna need pandas and PyYAML

```
$ pip install v20 pandas PyYAML
```
You can also use v20 for live trading (if your bot find golden nuggets!!!). The documentation [here](https://developer.oanda.com/rest-live-v20/introduction/) is quite comprehensive. 

But for now, let's write some python to get historical data. 

First, we read the config file `.v20.conf` from our `$HOME` path and create an API context. OANDA needs this to verify our accounts. 

Then, we write a function `get_hourly_candle` to get `n` data points from OANDA. I stick with hourly data, but you can change the granularity to download daily, monthly, or even 5-second data. Let's do `BAM` to get 'bid', 'ask', and 'mid' points.

Because OANDA allows maximum 5,000 bars per HTTP call, we need to write a loop to download data in chunks smaller than 5,000. The function `get_data` did just that. We then concatenate, reset index, and save data to a tab-delimited file.

{{< gist pipatth 2c04cf6e393388651cb4ac38dddd8000 "data.py" >}}

So if you call `get_data` function like below, you should get 30,000 bars of data
```
get_data("EUR_USD", "2014-01-01", 30000, "EUR_USD_train.tsv")
```
We also want to download some data to test the agent. This so-called 'testing' dataset should not be the same as the one we use to train the agent
```
get_data("EUR_USD", "2019-01-01", 2000, "EUR_USD_test.tsv")
```
So now we have some data. They should look something like this:
{{< gist pipatth 2c04cf6e393388651cb4ac38dddd8000 "EUR_USD_train.tsv" >}}

Now we're ready to build a trading environment. Part2 [here]({{< ref "/post/ppo2-currency-trader-part2.md" >}}).

Note: I used EUR_USD and GBP_USD as examples here. It's easier to work when you have a quote in your home currency. If your home currency is USD but you're trying to do USD_JPY or USD_CAD, just be careful.
