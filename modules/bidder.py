import numpy as np

#test
import streamlit as st

def buy_and_hold(init_val, symbol_closes):
    shares = init_val/symbol_closes[0]
    trade_val = shares*symbol_closes[-1]
    return trade_val

def buy_rule(init_value, transact_dates, transact_percents, dates, symbol_closes):
    close_values = [symbol_closes[dates.index(transact_date)] for transact_date in transact_dates]
    wallet = init_value
    shares = 0.0
    for close_value, transact_percent in zip(close_values, transact_percents):
        is_buy = transact_percent > 0
        if is_buy:
            transact_val = wallet*transact_percent/100
            transact_shares = transact_val/close_value
            wallet -= transact_val
            shares += transact_shares

            # st.write(f"buy: R{transact_val} = {transact_shares}shares. Remaining: R{wallet} and {shares}shares")
        else:
            transact_shares = shares*-transact_percent/100
            transact_val = transact_shares*close_value
            shares -= transact_shares
            wallet += transact_val

            # st.write(f"sell: R{transact_val} = {transact_shares}shares. Remaining: R{wallet} and {shares}shares")

    share_value = shares*symbol_closes[-1]
    return wallet + share_value