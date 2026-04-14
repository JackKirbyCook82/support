# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Concept Objects
@author: Jack Kirby Cook

"""

import numbers
from enum import Enum
from datetime import date as Date

from support.concepts import Assembly, Concepts, Concept, DateRange
from support.decorators import Dispatchers
from support.querys import Field, Query
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Concepts", "Querys", "Securities", "Strategies", "Alerting"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


AppraisalConcept = Concept("Appraisal", ["BLACKSCHOLES", "GREEKS"], start=1)
InstrumentConcept = Concept("Instrument", ["EMPTY", "STOCK", "OPTION"], start=0)
OptionConcept = Concept("Option", ["PUT", "EMPTY", "CALL"], start=-1)
PositionConcept = Concept("Position", ["SHORT", "EMPTY", "LONG"], start=-1)
SpreadConcept = Concept("Spread", ["EMPTY", "STRANGLE", "COLLAR", "VERTICAL"], start=0)
ValuationConcept = Concept("Valuation", ["ARBITRAGE", "RISKY", "WORTHLESS"], start=1)
ScenarioConcept = Concept("Scenario", ["MINIMUM", "MAXIMUM"], start=1)
StatusConcept = Concept("Status", ["PROSPECT", "PENDING", "OBSOLETE", "ABANDONED", "REJECTED", "ACCEPTED"], start=1)
TermConcept = Concept("Terms", ["MARKET", "LIMIT", "STOP", "STOPLIMIT", "LIMITDEBIT", "LIMITCREDIT"], start=1)
TenureConcept = Concept("Tenure", ["DAY", "STANDING", "OPENING", "CLOSING", "IMMEDIATE", "FILLKILL"], start=1)
PricingConcept = Concept("Pricing", ["AGGRESSIVE", "PASSIVE", "MODERATE"], start=1)
QuotingConcept = Concept("Quoting", ["LIVE", "FROZEN", "DELAYED"], start=1)
MarketConcept = Concept("Market", ["BEAR", "NEUTRAL", "BULL"], start=-1)
ActionConcept = Concept("Action", ["BUY", "SELL"], start=1)
StateConcept = Concept("State", ["BARS", "STATS"], start=1)
TrendConcept = Concept("Trend", ["SMA", "EMA", "MACD"], start=1)
MomentumConcept = Concept("Momentum", ["RSI"], start=1)
VolatilityConcept = Concept("Volatility", ["BB", "ATR"], start=1)
VolumeConcept = Concept("Volume", ["MFI", "CMF", "OBV"], start=1)

SecurityConcepts = Concepts("Security", ["instrument", "option", "position"], set())
StrategyConcepts = Concepts("Strategy", ["spread", "option", "position"], {"stocks", "options"})

TickerField = Field("ticker", str)
DateField = Field("date", Date, formatting="%Y%m%d")
ExpireField = Field("expire", Date, formatting="%Y%m%d")
StrikeField = Field("strike", numbers.Number, digits=2)
PriceField = Field("price", numbers.Number, digits=2)
AskField = Field("ask", numbers.Number, digits=2)
BidField = Field("bid", numbers.Number, digits=2)
InstrumentField = Field("instrument", Enum, variable=InstrumentConcept)
OptionField = Field("option", Enum, variable=OptionConcept)
PositionField = Field("position", Enum, variable=PositionConcept)

SymbolQuery = Query("Symbol", fields=[TickerField], bases=[], delimiter="|")
TradeQuery = Query("Trade", fields=[TickerField, PriceField], bases=[], delimiter="|")
QuoteQuery = Query("Quote", fields=[TickerField, BidField, AskField], bases=[], delimiter="|")
HistoryQuery = Query("History", fields=[TickerField, DateField], bases=[], delimiter="|")
SettlementQuery = Query("Settlement", fields=[TickerField, ExpireField], bases=[], delimiter="|")
ContractQuery = Query("Contract", fields=[TickerField, ExpireField, OptionField, StrikeField], bases=[], delimiter="|")

StockLongSecurity = SecurityConcepts("StockLong", [InstrumentConcept.STOCK, OptionConcept.EMPTY, PositionConcept.LONG])
StockShortSecurity = SecurityConcepts("StockShort", [InstrumentConcept.STOCK, OptionConcept.EMPTY, PositionConcept.SHORT])
OptionPutLongSecurity = SecurityConcepts("OptionPutLong", [InstrumentConcept.OPTION, OptionConcept.PUT, PositionConcept.LONG])
OptionPutShortSecurity = SecurityConcepts("OptionPutShort", [InstrumentConcept.OPTION, OptionConcept.PUT, PositionConcept.SHORT])
OptionCallLongSecurity = SecurityConcepts("OptionCallLong", [InstrumentConcept.OPTION, OptionConcept.CALL, PositionConcept.LONG])
OptionCallShortSecurity = SecurityConcepts("OptionCallShort", [InstrumentConcept.OPTION, OptionConcept.CALL, PositionConcept.SHORT])

VerticalPutStrategy = StrategyConcepts("VerticalPut", [SpreadConcept.VERTICAL, OptionConcept.PUT, PositionConcept.EMPTY], options=[OptionPutLongSecurity, OptionPutShortSecurity], stocks=[])
VerticalCallStrategy = StrategyConcepts("VerticalCall", [SpreadConcept.VERTICAL, OptionConcept.CALL, PositionConcept.EMPTY], options=[OptionCallLongSecurity, OptionCallShortSecurity], stocks=[])
CollarLongStrategy = StrategyConcepts("CollarLong", [SpreadConcept.COLLAR, OptionConcept.EMPTY, PositionConcept.LONG], options=[OptionPutLongSecurity, OptionCallShortSecurity], stocks=[StockLongSecurity])
CollarShortStrategy = StrategyConcepts("CollarShort", [SpreadConcept.COLLAR, OptionConcept.EMPTY, PositionConcept.SHORT], options=[OptionCallLongSecurity, OptionPutShortSecurity], stocks=[StockShortSecurity])


class Querys(Assembly): Symbol, Settlement, Contract = SymbolQuery, SettlementQuery, ContractQuery
class Concepts(Assembly):
    class Securities(Assembly): Security, Instrument, Option, Position = SecurityConcepts, InstrumentConcept, OptionConcept, PositionConcept
    class Strategies(Assembly): Strategy, Spread = StrategyConcepts, SpreadConcept
    class Markets(Assembly): Status, Term, Tenure, Action, Quoting = StatusConcept, TermConcept, TenureConcept, ActionConcept, QuotingConcept
    class Technicals(Assembly): State, Trend, Momentum, Volatility, Volume = StateConcept, TrendConcept, MomentumConcept, VolatilityConcept, VolumeConcept
    Pricing = PricingConcept
    Appraisal = AppraisalConcept
    Valuation = ValuationConcept
    Scenario = ScenarioConcept
    Market = MarketConcept

class Securities(Assembly):
    class Stocks(Assembly): Long = StockLongSecurity; Short = StockShortSecurity
    class Options(Assembly):
        class Puts(Assembly): Long = OptionPutLongSecurity; Short = OptionPutShortSecurity
        class Calls(Assembly): Long = OptionCallLongSecurity; Short = OptionCallShortSecurity

class Strategies(Assembly):
    class Verticals(Assembly): Put = VerticalPutStrategy; Call = VerticalCallStrategy
    class Collars(Assembly): Long = CollarLongStrategy; Short = CollarShortStrategy


class Alerting(Logging):
    @Dispatchers.Value(locator="instrument")
    def alert(self, dataframe, *args, instrument, **kwargs): raise ValueError(instrument)

    @alert.register(Concepts.Securities.Instrument.STOCK)
    def stock(self, dataframe, *args, instrument, **kwargs):
        tickers = "|".join(list(dataframe["ticker"].unique()))
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(dataframe.index))
        sizes = f"{int(previous):.0f}|{int(post):.0f}" if previous is not None else f"{len(dataframe.index):.0f}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(sizes)}]")

    @alert.register(Concepts.Securities.Instrument.OPTION)
    def option(self, dataframe, *args, instrument, **kwargs):
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(dataframe.index))
        sizes = f"{int(previous):.0f}|{int(post):.0f}" if previous is not None else f"{len(dataframe.index):.0f}"
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {str(sizes)}]")



