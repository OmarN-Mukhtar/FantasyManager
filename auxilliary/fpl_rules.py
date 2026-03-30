"""
Fantasy Premier League Rules and Constraints
"""

FPL_RULES = """
# Fantasy Premier League Team Selection Rules

## Budget and Squad Size
- Total budget: £100.0 million
- Squad size: 15 players total
  * 2 Goalkeepers (GK)
  * 5 Defenders (DEF)
  * 5 Midfielders (MID)
  * 3 Forwards (FWD)

## Team Limits
- Maximum 3 players from any single Premier League team
- Cannot have more than 3 players from the same club

## Starting XI Formation
- Must select 11 players from your 15-player squad
- Must include:
  * Exactly 1 Goalkeeper
  * At least 3 Defenders
  * At least 2 Midfielders  
  * At least 1 Forward
- Valid formations:
  * 3-4-3 (3 DEF, 4 MID, 3 FWD)
  * 3-5-2 (3 DEF, 5 MID, 2 FWD)
  * 4-3-3 (4 DEF, 3 MID, 3 FWD)
  * 4-4-2 (4 DEF, 4 MID, 2 FWD)
  * 4-5-1 (4 DEF, 5 MID, 1 FWD)
  * 5-3-2 (5 DEF, 3 MID, 2 FWD)
  * 5-4-1 (5 DEF, 4 MID, 1 FWD)

## Captain and Vice-Captain
- Select 1 Captain: receives double points
- Select 1 Vice-Captain: receives double points if Captain doesn't play

## Transfers
- 1 free transfer per gameweek
- Additional transfers cost 4 points each
- Unused free transfers roll over (max 2 free transfers)
- Wildcard: make unlimited free transfers (2 per season)

## Chips (one-time use each season)
- Bench Boost: Get points from all bench players for one gameweek
- Triple Captain: Captain gets triple points instead of double
- Free Hit: Make unlimited transfers for one gameweek, team reverts after
- Wildcard: Unlimited free transfers (available twice per season)

## Points Scoring System

### All Players
- Playing 0-60 minutes: 1 point
- Playing 60+ minutes: 2 points
- Yellow card: -1 point
- Red card: -3 points

### Goalkeepers & Defenders
- Clean sheet (no goals conceded while playing 60+ mins): 4 points (GK/DEF)
- Goal scored: 6 points (GK/DEF)
- Assist: 3 points
- Save (GK only): 1 point per 3 saves
- Penalty save (GK only): 5 points
- Penalty miss: -2 points
- Own goal: -2 points
- Conceding 2+ goals: -1 point per goal

### Midfielders
- Clean sheet: 1 point
- Goal scored: 5 points
- Assist: 3 points
- Penalty miss: -2 points
- Own goal: -2 points

### Forwards
- Goal scored: 4 points
- Assist: 3 points
- Penalty miss: -2 points
- Own goal: -2 points

### Bonus Points (BPS)
- Top 3 players in each match get bonus points
- 1st place: 3 bonus points
- 2nd place: 2 bonus points
- 3rd place: 1 bonus point

## Strategy Considerations
- Look for players with good upcoming fixtures (easy opponents)
- Consider form over the last 5-10 gameweeks
- Balance premium players (expensive) with budget options (value picks)
- Defenders from teams with good defensive records get clean sheet points
- Attacking players (goals/assists) generally outscore others
- Midfielders who play as forwards offer best value (MID price, FWD returns)
- Rotation risk: avoid players who don't play regularly
- Injury prone players are risky despite talent
- Monitor pre-match press conferences for team news
"""

# Position constraints
SQUAD_REQUIREMENTS = {
    'GK': 2,
    'DEF': 5,
    'MID': 5,
    'FWD': 3
}

STARTING_XI_MIN = {
    'GK': 1,
    'DEF': 3,
    'MID': 2,
    'FWD': 1
}

STARTING_XI_MAX = {
    'GK': 1,
    'DEF': 5,
    'MID': 5,
    'FWD': 3
}

BUDGET = 100.0  # Million
MAX_PLAYERS_PER_TEAM = 3
SQUAD_SIZE = 15
STARTING_XI_SIZE = 11


def validate_squad(players_df):
    """
    Validate if a squad meets FPL requirements.
    
    Args:
        players_df: DataFrame with columns ['position', 'team', 'now_cost']
    
    Returns:
        dict with validation results
    """
    issues = []
    
    # Check squad size
    if len(players_df) != SQUAD_SIZE:
        issues.append(f"Squad must have exactly {SQUAD_SIZE} players (currently {len(players_df)})")
    
    # Check position requirements
    position_counts = players_df['position'].value_counts().to_dict()
    for pos, required in SQUAD_REQUIREMENTS.items():
        actual = position_counts.get(pos, 0)
        if actual != required:
            issues.append(f"Need {required} {pos} (currently {actual})")
    
    # Check budget
    total_cost = players_df['now_cost'].sum()
    if total_cost > BUDGET:
        issues.append(f"Over budget: £{total_cost:.1f}M (max £{BUDGET}M)")
    
    # Check team limits
    team_counts = players_df['team'].value_counts()
    over_limit = team_counts[team_counts > MAX_PLAYERS_PER_TEAM]
    if not over_limit.empty:
        for team, count in over_limit.items():
            issues.append(f"Too many players from {team}: {count} (max {MAX_PLAYERS_PER_TEAM})")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_cost': total_cost,
        'remaining_budget': BUDGET - total_cost
    }


def suggest_formation(starting_xi_df):
    """Suggest valid formation for starting XI."""
    position_counts = starting_xi_df['position'].value_counts().to_dict()
    gk = position_counts.get('GK', 0)
    def_ = position_counts.get('DEF', 0)
    mid = position_counts.get('MID', 0)
    fwd = position_counts.get('FWD', 0)
    
    return f"{gk}-{def_}-{mid}-{fwd}"
