import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PlayerStats:
    """
    Class to store and analyze player statistics for the Momentum Shift Score (MSS) project.
    Handles both batters and pitchers, with different calculation methods for each.
    """
    
    def __init__(self, player_name, player_id, stats_df, start_date, end_date, player_type='batter'):
        """
        Initialize a PlayerStats object with the player's statistics.
        
        Parameters:
        -----------
        player_name : str
            The name of the player
        player_id : int
            The MLB ID of the player
        stats_df : pandas.DataFrame
            DataFrame containing the player's statistics
        start_date : str
            The start date of the statistics period (YYYY-MM-DD)
        end_date : str
            The end date of the statistics period (YYYY-MM-DD)
        player_type : str
            Either 'batter' or 'pitcher' to indicate the type of player
        """
        self.player_name = player_name
        self.player_id = player_id
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = stats_df
        self.player_type = player_type.lower()
        self.data_available = not stats_df.empty
        
        # Initialize containers for calculated statistics
        self.summary_stats = {}
        self.pitch_type_stats = {}
        self.situational_stats = {}
        self.batted_ball_profile = {}
        
        # Calculate derived statistics if data is available
        if self.data_available:
            self._calculate_all_stats()
    
    def _calculate_all_stats(self):
        """Calculate all statistics based on player type"""
        if self.player_type == 'batter':
            self._calculate_batter_stats()
        elif self.player_type == 'pitcher':
            self._calculate_pitcher_stats()
        else:
            raise ValueError(f"Unknown player type: {self.player_type}. Must be 'batter' or 'pitcher'.")
    
    def _calculate_batter_stats(self):
        """Calculate statistics for batters"""
        self._calculate_batter_summary_stats()
        self._calculate_batter_batted_ball_profile()
        self._calculate_batter_pitch_type_stats()
        self._calculate_batter_situational_stats()
    
    def _calculate_pitcher_stats(self):
        """Calculate statistics for pitchers"""
        self._calculate_pitcher_summary_stats()
        self._calculate_pitcher_batted_ball_profile()
        self._calculate_pitcher_pitch_type_stats()
        self._calculate_pitcher_situational_stats()
    
    def _calculate_batter_summary_stats(self):
        """Calculate overall performance metrics for batters"""
        stats = self.raw_data
        
        # Count events by type
        events_counts = stats['events'].value_counts().to_dict() if 'events' in stats.columns else {}
        
        # Calculate basic statistics
        total_pas = sum(pd.notna(stats['events'])) if 'events' in stats.columns else len(stats)
        
        # For at-bats, we need to exclude walks, HBP, etc.
        if 'events' in stats.columns:
            total_abs = sum(pd.notna(stats['events']) & ~stats['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']))
        else:
            total_abs = total_pas  # Fallback if events column is missing
        
        # Count hits by type
        singles = events_counts.get('single', 0)
        doubles = events_counts.get('double', 0)
        triples = events_counts.get('triple', 0)
        home_runs = events_counts.get('home_run', 0)
        hits = singles + doubles + triples + home_runs
        
        # Calculate other offensive stats
        walks = events_counts.get('walk', 0)
        strikeouts = events_counts.get('strikeout', 0)
        hit_by_pitch = events_counts.get('hit_by_pitch', 0)
        
        # Derived statistics
        batting_avg = hits / total_abs if total_abs > 0 else 0
        on_base_pct = (hits + walks + hit_by_pitch) / total_pas if total_pas > 0 else 0
        slug_pct = (singles + 2*doubles + 3*triples + 4*home_runs) / total_abs if total_abs > 0 else 0
        ops = on_base_pct + slug_pct
        
        # Advanced metrics
        woba_value = stats['woba_value'].sum() if 'woba_value' in stats.columns else 0
        woba_denom = stats['woba_denom'].sum() if 'woba_denom' in stats.columns else 0
        woba = woba_value / woba_denom if woba_denom > 0 else 0
        
        # Statcast metrics
        avg_launch_speed = stats['launch_speed'].dropna().mean() if 'launch_speed' in stats.columns and not stats['launch_speed'].dropna().empty else 0
        avg_launch_angle = stats['launch_angle'].dropna().mean() if 'launch_angle' in stats.columns and not stats['launch_angle'].dropna().empty else 0
        max_exit_velo = stats['launch_speed'].max() if 'launch_speed' in stats.columns and not stats['launch_speed'].dropna().empty else 0
        max_distance = stats['hit_distance_sc'].max() if 'hit_distance_sc' in stats.columns and not stats['hit_distance_sc'].dropna().empty else 0
        
        # Store all calculated stats in the summary_stats dictionary
        self.summary_stats = {
            'total_pas': total_pas,
            'total_abs': total_abs,
            'singles': singles,
            'doubles': doubles,
            'triples': triples,
            'home_runs': home_runs,
            'hits': hits,
            'walks': walks,
            'strikeouts': strikeouts,
            'hit_by_pitch': hit_by_pitch,
            'batting_avg': batting_avg,
            'on_base_pct': on_base_pct,
            'slug_pct': slug_pct,
            'ops': ops,
            'woba': woba,
            'avg_launch_speed': avg_launch_speed,
            'avg_launch_angle': avg_launch_angle,
            'max_exit_velo': max_exit_velo,
            'max_distance': max_distance,
            'events_counts': events_counts
        }
    
    def _calculate_batter_batted_ball_profile(self):
        """Calculate batted ball profile for batters"""
        stats = self.raw_data
        
        # Filter to only include batted balls
        batted_balls = stats[pd.notna(stats['bb_type'])] if 'bb_type' in stats.columns else pd.DataFrame()
        
        if batted_balls.empty:
            self.batted_ball_profile = {'available': False}
            return
        
        # Count batted ball types
        bb_types = batted_balls['bb_type'].value_counts().to_dict()
        total_batted_balls = len(batted_balls)
        
        # Calculate percentages
        gb_pct = bb_types.get('ground_ball', 0) / total_batted_balls if total_batted_balls > 0 else 0
        fb_pct = bb_types.get('fly_ball', 0) / total_batted_balls if total_batted_balls > 0 else 0
        ld_pct = bb_types.get('line_drive', 0) / total_batted_balls if total_batted_balls > 0 else 0
        pu_pct = bb_types.get('popup', 0) / total_batted_balls if total_batted_balls > 0 else 0
        
        # Calculate hard hit rate (95+ mph)
        hard_hit_balls = len(batted_balls[batted_balls['launch_speed'] >= 95]) if 'launch_speed' in batted_balls.columns else 0
        hard_hit_rate = hard_hit_balls / total_batted_balls if total_batted_balls > 0 else 0
        
        # Calculate barrel rate
        # Barrels require minimum exit velocity of 98 mph and a launch angle between 8 and 32 degrees
        barrels = 0
        if 'launch_speed' in batted_balls.columns and 'launch_angle' in batted_balls.columns:
            barrels = len(batted_balls[(batted_balls['launch_speed'] >= 98) & 
                                      (batted_balls['launch_angle'] >= 8) & 
                                      (batted_balls['launch_angle'] <= 32)])
        barrel_rate = barrels / total_batted_balls if total_batted_balls > 0 else 0
        
        # Store the results
        self.batted_ball_profile = {
            'available': True,
            'ground_ball_pct': gb_pct,
            'fly_ball_pct': fb_pct,
            'line_drive_pct': ld_pct,
            'popup_pct': pu_pct,
            'hard_hit_rate': hard_hit_rate,
            'barrel_rate': barrel_rate,
            'total_batted_balls': total_batted_balls,
            'batted_ball_types': bb_types
        }
    
    def _calculate_batter_pitch_type_stats(self):
        """Calculate pitch type statistics for batters"""
        stats = self.raw_data
        
        # Check if pitch type data is available
        if 'pitch_type' not in stats.columns:
            self.pitch_type_stats = {'available': False}
            return
        
        # Group by pitch type
        pitch_groups = stats.groupby('pitch_type')
        pitch_stats = {}
        
        for pitch_type, group in pitch_groups:
            if pd.isna(pitch_type) or pitch_type == '':
                continue
                
            # Count outcomes
            outcomes = group['description'].value_counts().to_dict() if 'description' in group.columns else {}
            
            # Calculate metrics
            total_pitches = len(group)
            strikes = sum(group['description'].isin(['called_strike', 'swinging_strike', 'foul', 'foul_tip'])) if 'description' in group.columns else 0
            balls = sum(group['description'] == 'ball') if 'description' in group.columns else 0
            swing_count = sum(group['description'].isin(['swinging_strike', 'foul', 'foul_tip', 'hit_into_play'])) if 'description' in group.columns else 0
            whiff_count = sum(group['description'] == 'swinging_strike') if 'description' in group.columns else 0
            
            # Calculate rates
            strike_rate = strikes / total_pitches if total_pitches > 0 else 0
            swing_rate = swing_count / total_pitches if total_pitches > 0 else 0
            whiff_rate = whiff_count / swing_count if swing_count > 0 else 0
            
            # Velocity and spin data
            avg_velo = group['release_speed'].mean() if 'release_speed' in group.columns and not group['release_speed'].dropna().empty else 0
            avg_spin = group['release_spin_rate'].mean() if 'release_spin_rate' in group.columns and not group['release_spin_rate'].dropna().empty else 0
            
            # Events resulting from this pitch type
            events = group['events'].value_counts().to_dict() if 'events' in group.columns else {}
            
            # Contact quality
            contact_results = {}
            if 'launch_speed' in group.columns and 'launch_angle' in group.columns:
                contact_balls = group[group['description'] == 'hit_into_play']
                if not contact_balls.empty:
                    contact_results = {
                        'avg_exit_velo': contact_balls['launch_speed'].mean() if not contact_balls['launch_speed'].dropna().empty else 0,
                        'avg_launch_angle': contact_balls['launch_angle'].mean() if not contact_balls['launch_angle'].dropna().empty else 0
                    }
            
            # Store the stats for this pitch type
            pitch_stats[pitch_type] = {
                'total_pitches': total_pitches,
                'strike_rate': strike_rate,
                'swing_rate': swing_rate,
                'whiff_rate': whiff_rate,
                'avg_velocity': avg_velo,
                'avg_spin_rate': avg_spin,
                'outcomes': outcomes,
                'events': events,
                'contact_results': contact_results
            }
        
        self.pitch_type_stats = {
            'available': len(pitch_stats) > 0,
            'pitch_types': pitch_stats
        }
    
    def _calculate_batter_situational_stats(self):
        """Calculate situational statistics for batters"""
        stats = self.raw_data
        
        # Initialize container for situational stats
        situational = {}
        
        # By count
        if 'balls' in stats.columns and 'strikes' in stats.columns:
            # Define key counts
            ahead_counts = stats[(stats['balls'] > stats['strikes'])]
            behind_counts = stats[(stats['balls'] < stats['strikes'])]
            even_counts = stats[(stats['balls'] == stats['strikes']) & (stats['balls'] > 0)]
            
            # Calculate performance in different counts
            for name, group in [('ahead_count', ahead_counts), 
                                ('behind_count', behind_counts), 
                                ('even_count', even_counts)]:
                
                if not group.empty and 'events' in group.columns:
                    hits = sum(pd.notna(group['events']) & group['events'].isin(['single', 'double', 'triple', 'home_run']))
                    at_bats = sum(pd.notna(group['events']) & ~group['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']))
                    
                    situational[name] = {
                        'batting_avg': hits / at_bats if at_bats > 0 else 0,
                        'total_at_bats': at_bats
                    }
        
        # By men on base situation
        if all(col in stats.columns for col in ['on_1b', 'on_2b', 'on_3b']):
            # Define different base states
            empty = stats[(stats['on_1b'].isna()) & (stats['on_2b'].isna()) & (stats['on_3b'].isna())]
            risp = stats[(~stats['on_2b'].isna()) | (~stats['on_3b'].isna())]
            
            # Calculate performance in different base states
            for name, group in [('bases_empty', empty), ('risp', risp)]:
                if not group.empty and 'events' in group.columns:
                    hits = sum(pd.notna(group['events']) & group['events'].isin(['single', 'double', 'triple', 'home_run']))
                    at_bats = sum(pd.notna(group['events']) & ~group['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']))
                    
                    situational[name] = {
                        'batting_avg': hits / at_bats if at_bats > 0 else 0,
                        'total_at_bats': at_bats
                    }
        
        # By inning (early/middle/late)
        if 'inning' in stats.columns:
            early = stats[stats['inning'] <= 3]
            middle = stats[(stats['inning'] > 3) & (stats['inning'] <= 6)]
            late = stats[stats['inning'] > 6]
            
            # Calculate performance in different game stages
            for name, group in [('early_innings', early), 
                                ('middle_innings', middle), 
                                ('late_innings', late)]:
                
                if not group.empty and 'events' in group.columns:
                    hits = sum(pd.notna(group['events']) & group['events'].isin(['single', 'double', 'triple', 'home_run']))
                    at_bats = sum(pd.notna(group['events']) & ~group['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']))
                    
                    situational[name] = {
                        'batting_avg': hits / at_bats if at_bats > 0 else 0,
                        'total_at_bats': at_bats
                    }
        
        self.situational_stats = situational
    
    def _calculate_pitcher_summary_stats(self):
        """Calculate overall performance metrics for pitchers"""
        stats = self.raw_data
        
        # Count events by type
        events_counts = stats['events'].value_counts().to_dict() if 'events' in stats.columns else {}
        
        # Calculate basic statistics
        total_batters_faced = stats['at_bat_number'].nunique() if 'at_bat_number' in stats.columns else len(stats)
        total_pitches = len(stats)
        
        # Count events
        hits = sum(pd.notna(stats['events']) & stats['events'].isin(['single', 'double', 'triple', 'home_run'])) if 'events' in stats.columns else 0
        walks = events_counts.get('walk', 0)
        strikeouts = events_counts.get('strikeout', 0)
        home_runs = events_counts.get('home_run', 0)
        
        # Derived statistics - need to estimate innings pitched
        # For simplicity, assume 3 outs = 1 inning
        field_outs = events_counts.get('field_out', 0)
        force_outs = events_counts.get('force_out', 0)
        double_plays = events_counts.get('grounded_into_double_play', 0) * 2
        outs = field_outs + strikeouts + force_outs + double_plays
        innings_pitched = outs / 3 if outs > 0 else total_batters_faced / 3
        
        # Calculate pitching metrics - avoiding negative ERA due to calculation issues
        runs_allowed = sum(pd.notna(stats['runs_scored_on_play']) & (stats['runs_scored_on_play'] > 0)) if 'runs_scored_on_play' in stats.columns else hits * 0.5
        
        # Calculate ERA - ensure it's not negative
        era = (9 * runs_allowed) / innings_pitched if innings_pitched > 0 else 0
        era = max(0, era)  # Ensure ERA is not negative
        
        whip = (hits + walks) / innings_pitched if innings_pitched > 0 else 0
        k_per_9 = (9 * strikeouts) / innings_pitched if innings_pitched > 0 else 0
        bb_per_9 = (9 * walks) / innings_pitched if innings_pitched > 0 else 0
        hr_per_9 = (9 * home_runs) / innings_pitched if innings_pitched > 0 else 0
        k_bb_ratio = strikeouts / walks if walks > 0 else strikeouts  # Return strikeouts instead of inf
        
        # Statcast metrics
        avg_velo = stats['release_speed'].mean() if 'release_speed' in stats.columns and not stats['release_speed'].dropna().empty else 0
        avg_spin = stats['release_spin_rate'].mean() if 'release_spin_rate' in stats.columns and not stats['release_spin_rate'].dropna().empty else 0
        
        # Store all calculated stats
        self.summary_stats = {
            'total_batters_faced': total_batters_faced,
            'total_pitches': total_pitches,
            'innings_pitched': innings_pitched,
            'hits': hits,
            'walks': walks,
            'strikeouts': strikeouts,
            'home_runs': home_runs,
            'runs_allowed': runs_allowed,
            'era': era,
            'whip': whip,
            'k_per_9': k_per_9,
            'bb_per_9': bb_per_9,
            'hr_per_9': hr_per_9,
            'k_bb_ratio': k_bb_ratio,
            'avg_velocity': avg_velo,
            'avg_spin_rate': avg_spin,
            'events_counts': events_counts
        }
    
    def _calculate_pitcher_batted_ball_profile(self):
        """Calculate batted ball profile for pitchers"""
        stats = self.raw_data
        
        # Filter to only include batted balls
        batted_balls = stats[pd.notna(stats['bb_type'])] if 'bb_type' in stats.columns else pd.DataFrame()
        
        if batted_balls.empty:
            self.batted_ball_profile = {'available': False}
            return
        
        # Count batted ball types
        bb_types = batted_balls['bb_type'].value_counts().to_dict()
        total_batted_balls = len(batted_balls)
        
        # Calculate percentages
        gb_pct = bb_types.get('ground_ball', 0) / total_batted_balls if total_batted_balls > 0 else 0
        fb_pct = bb_types.get('fly_ball', 0) / total_batted_balls if total_batted_balls > 0 else 0
        ld_pct = bb_types.get('line_drive', 0) / total_batted_balls if total_batted_balls > 0 else 0
        pu_pct = bb_types.get('popup', 0) / total_batted_balls if total_batted_balls > 0 else 0
        
        # Calculate hard hit rate allowed (95+ mph)
        hard_hit_balls = len(batted_balls[batted_balls['launch_speed'] >= 95]) if 'launch_speed' in batted_balls.columns else 0
        hard_hit_rate = hard_hit_balls / total_batted_balls if total_batted_balls > 0 else 0
        
        # Calculate barrel rate allowed
        barrels = 0
        if 'launch_speed' in batted_balls.columns and 'launch_angle' in batted_balls.columns:
            barrels = len(batted_balls[(batted_balls['launch_speed'] >= 98) & 
                                      (batted_balls['launch_angle'] >= 8) & 
                                      (batted_balls['launch_angle'] <= 32)])
        barrel_rate = barrels / total_batted_balls if total_batted_balls > 0 else 0
        
        # Store the results
        self.batted_ball_profile = {
            'available': True,
            'ground_ball_pct': gb_pct,
            'fly_ball_pct': fb_pct,
            'line_drive_pct': ld_pct,
            'popup_pct': pu_pct,
            'hard_hit_rate_allowed': hard_hit_rate,
            'barrel_rate_allowed': barrel_rate,
            'total_batted_balls': total_batted_balls,
            'batted_ball_types': bb_types
        }
    
    def _calculate_pitcher_pitch_type_stats(self):
        """Calculate pitch type statistics for pitchers"""
        stats = self.raw_data
        
        # Check if pitch type data is available
        if 'pitch_type' not in stats.columns:
            self.pitch_type_stats = {'available': False}
            return
        
        # Group by pitch type
        pitch_groups = stats.groupby('pitch_type')
        pitch_stats = {}
        
        for pitch_type, group in pitch_groups:
            if pd.isna(pitch_type) or pitch_type == '':
                continue
                
            # Count outcomes
            outcomes = group['description'].value_counts().to_dict() if 'description' in group.columns else {}
            
            # Calculate metrics
            total_pitches = len(group)
            strikes = sum(group['description'].isin(['called_strike', 'swinging_strike', 'foul', 'foul_tip'])) if 'description' in group.columns else 0
            balls = sum(group['description'] == 'ball') if 'description' in group.columns else 0
            swing_count = sum(group['description'].isin(['swinging_strike', 'foul', 'foul_tip', 'hit_into_play'])) if 'description' in group.columns else 0
            whiff_count = sum(group['description'] == 'swinging_strike') if 'description' in group.columns else 0
            in_play = sum(group['description'] == 'hit_into_play') if 'description' in group.columns else 0
            
            # Calculate rates
            strike_rate = strikes / total_pitches if total_pitches > 0 else 0
            swing_rate = swing_count / total_pitches if total_pitches > 0 else 0
            whiff_rate = whiff_count / swing_count if swing_count > 0 else 0
            in_play_rate = in_play / swing_count if swing_count > 0 else 0
            
            # Velocity and spin data
            avg_velo = group['release_speed'].mean() if 'release_speed' in group.columns and not group['release_speed'].dropna().empty else 0
            avg_spin = group['release_spin_rate'].mean() if 'release_spin_rate' in group.columns and not group['release_spin_rate'].dropna().empty else 0
            
            # Events resulting from this pitch type
            events = group['events'].value_counts().to_dict() if 'events' in group.columns else {}
            
            # Contact quality
            contact_results = {}
            if 'launch_speed' in group.columns and 'launch_angle' in group.columns:
                contact_balls = group[group['description'] == 'hit_into_play']
                if not contact_balls.empty:
                    contact_results = {
                        'avg_exit_velo': contact_balls['launch_speed'].mean() if not contact_balls['launch_speed'].dropna().empty else 0,
                        'avg_launch_angle': contact_balls['launch_angle'].mean() if not contact_balls['launch_angle'].dropna().empty else 0
                    }
            
            # Calculate put_away_rate safely
            put_away_rate = 0
            if 'description' in group.columns:
                total_put_away_opps = sum(group['description'].isin(['swinging_strike', 'hit_into_play']))
                if total_put_away_opps > 0:
                    put_away_rate = sum(group['description'] == 'swinging_strike') / total_put_away_opps
            
            # Store the stats for this pitch type
            pitch_stats[pitch_type] = {
                'total_pitches': total_pitches,
                'strike_rate': strike_rate,
                'swing_rate': swing_rate,
                'whiff_rate': whiff_rate,
                'in_play_rate': in_play_rate,
                'put_away_rate': put_away_rate,
                'avg_velocity': avg_velo,
                'avg_spin_rate': avg_spin,
                'events': events,
                'contact_results': contact_results
            }
        
        self.pitch_type_stats = {
            'available': len(pitch_stats) > 0,
            'pitch_types': pitch_stats
        }
    
    def _calculate_pitcher_situational_stats(self):
        """Calculate situational statistics for pitchers"""
        stats = self.raw_data
        
        # Initialize container for situational stats
        situational = {}
        
        # By count
        if 'balls' in stats.columns and 'strikes' in stats.columns:
            # Define key counts
            ahead_counts = stats[(stats['balls'] < stats['strikes'])]  # Reversed for pitchers
            behind_counts = stats[(stats['balls'] > stats['strikes'])]  # Reversed for pitchers
            even_counts = stats[(stats['balls'] == stats['strikes']) & (stats['balls'] > 0)]
            
            # Calculate performance in different counts
            for name, group in [('ahead_count', ahead_counts), 
                                ('behind_count', behind_counts), 
                                ('even_count', even_counts)]:
                
                if not group.empty and 'events' in group.columns:
                    hits = sum(pd.notna(group['events']) & group['events'].isin(['single', 'double', 'triple', 'home_run']))
                    at_bats = sum(pd.notna(group['events']) & ~group['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']))
                    
                    situational[name] = {
                        'batting_avg_against': hits / at_bats if at_bats > 0 else 0,
                        'total_at_bats': at_bats
                    }
        
        # By men on base situation
        if all(col in stats.columns for col in ['on_1b', 'on_2b', 'on_3b']):
            # Define different base states
            empty = stats[(stats['on_1b'].isna()) & (stats['on_2b'].isna()) & (stats['on_3b'].isna())]
            risp = stats[(~stats['on_2b'].isna()) | (~stats['on_3b'].isna())]
            
            # Calculate performance in different base states
            for name, group in [('bases_empty', empty), ('risp', risp)]:
                if not group.empty and 'events' in group.columns:
                    hits = sum(pd.notna(group['events']) & group['events'].isin(['single', 'double', 'triple', 'home_run']))
                    at_bats = sum(pd.notna(group['events']) & ~group['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']))
                    
                    situational[name] = {
                        'batting_avg_against': hits / at_bats if at_bats > 0 else 0,
                        'total_at_bats': at_bats
                    }
        
        # By inning (early/middle/late)
        if 'inning' in stats.columns:
            early = stats[stats['inning'] <= 3]
            middle = stats[(stats['inning'] > 3) & (stats['inning'] <= 6)]
            late = stats[stats['inning'] > 6]
            
            # Calculate performance in different game stages
            for name, group in [('early_innings', early), 
                                ('middle_innings', middle), 
                                ('late_innings', late)]:
                
                if not group.empty and 'events' in group.columns:
                    hits = sum(pd.notna(group['events']) & group['events'].isin(['single', 'double', 'triple', 'home_run']))
                    at_bats = sum(pd.notna(group['events']) & ~group['events'].isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt']))
                    
                    situational[name] = {
                        'batting_avg_against': hits / at_bats if at_bats > 0 else 0,
                        'total_at_bats': at_bats
                    }
        
        self.situational_stats = situational
    
    def get_summary(self):
        """Return a dictionary summary of key stats"""
        if not self.data_available:
            return {
                'player_name': self.player_name,
                'player_id': self.player_id,
                'data_available': False
            }
        
        # Different summary formats based on player type
        if self.player_type == 'batter':
            return {
                'player_name': self.player_name,
                'player_id': self.player_id,
                'data_available': True,
                'date_range': {
                    'start': self.start_date,
                    'end': self.end_date
                },
                'batting_avg': self.summary_stats.get('batting_avg', 0),
                'on_base_pct': self.summary_stats.get('on_base_pct', 0),
                'slug_pct': self.summary_stats.get('slug_pct', 0),
                'ops': self.summary_stats.get('ops', 0),
                'home_runs': self.summary_stats.get('home_runs', 0),
                'total_abs': self.summary_stats.get('total_abs', 0),
                'strikeout_rate': self.summary_stats.get('strikeouts', 0) / max(self.summary_stats.get('total_pas', 1), 1),
                'walk_rate': self.summary_stats.get('walks', 0) / max(self.summary_stats.get('total_pas', 1), 1),
                'avg_launch_speed': self.summary_stats.get('avg_launch_speed', 0),
                'avg_launch_angle': self.summary_stats.get('avg_launch_angle', 0),
                'barrel_rate': self.batted_ball_profile.get('barrel_rate', 0) if self.batted_ball_profile.get('available', False) else 0
            }
        else:  # pitcher
            return {
                'player_name': self.player_name,
                'player_id': self.player_id,
                'data_available': True,
                'date_range': {
                    'start': self.start_date,
                    'end': self.end_date
                },
                'innings_pitched': self.summary_stats.get('innings_pitched', 0),
                'era': self.summary_stats.get('era', 0),  # Changed from float('inf') to 0
                'whip': self.summary_stats.get('whip', 0),  # Changed from float('inf') to 0
                'k_per_9': self.summary_stats.get('k_per_9', 0),
                'bb_per_9': self.summary_stats.get('bb_per_9', 0),
                'hr_per_9': self.summary_stats.get('hr_per_9', 0),
                'k_bb_ratio': self.summary_stats.get('k_bb_ratio', 0),  # Changed from float('inf') to 0
                'avg_velocity': self.summary_stats.get('avg_velocity', 0),
                'ground_ball_pct': self.batted_ball_profile.get('ground_ball_pct', 0) if self.batted_ball_profile.get('available', False) else 0,
                'barrel_rate_allowed': self.batted_ball_profile.get('barrel_rate_allowed', 0) if self.batted_ball_profile.get('available', False) else 0
            }
    
    def compare_to(self, other_stats):
        """
        Compare this player's stats to another PlayerStats object.
        Returns a dictionary with the differences in key metrics.
        
        Parameters:
        -----------
        other_stats : PlayerStats
            Another PlayerStats object to compare against
        
        Returns:
        --------
        dict
            Dictionary of key metrics and their differences
        """
        if not self.data_available or not other_stats.data_available:
            return {'comparison_available': False}
        
        if self.player_type != other_stats.player_type:
            return {'comparison_available': False, 'error': 'Cannot compare different player types'}
        
        # Get summaries for both objects
        self_summary = self.get_summary()
        other_summary = other_stats.get_summary()
        
        # Initialize results dictionary
        comparison = {
            'comparison_available': True,
            'player_name': self.player_name,
            'player_id': self.player_id,
            'from_date': self.start_date,
            'to_date': other_stats.end_date,
            'comparison_period': {
                'before': {
                    'start': self.start_date,
                    'end': self.end_date
                },
                'after': {
                    'start': other_stats.start_date,
                    'end': other_stats.end_date
                }
            }
        }
        
        # Add differences for common metrics
        for key in self_summary:
            if key in ('player_name', 'player_id', 'data_available', 'date_range'):
                continue
                
            if key in other_summary:
                comparison[f'{key}_change'] = other_summary[key] - self_summary[key]
                comparison[f'{key}_pct_change'] = ((other_summary[key] / self_summary[key]) - 1) * 100 if self_summary[key] != 0 else 0
        
        print(comparison)
        return comparison
    
    def calculate_mss(self, other_stats, weight_config=None):
        """
        Calculate Momentum Shift Score by comparing this player's stats to another PlayerStats object.
        
        Parameters:
        -----------
        other_stats : PlayerStats
            Another PlayerStats object to compare against (typically post-moment stats)
        weight_config : dict, optional
            Configuration for weighting different components of the MSS calculation
            
        Returns:
        --------
        float
            The calculated Momentum Shift Score
        dict
            Detailed breakdown of the MSS calculation
        """
        # Default weights if not provided
        if weight_config is None:
            if self.player_type == 'batter':
                weight_config = {
                    'batting_avg': 0.15,
                    'on_base_pct': 0.15,
                    'slug_pct': 0.15,
                    'home_runs_rate': 0.1,
                    'strikeout_rate': 0.1,
                    'barrel_rate': 0.15,
                    'launch_speed': 0.1,
                    'situational': 0.1,
                }
            else:  # pitcher
                weight_config = {
                    'era': 0.15,
                    'whip': 0.15,
                    'k_per_9': 0.15,
                    'bb_per_9': 0.1,
                    'hr_per_9': 0.1,
                    'barrel_rate': 0.15,
                    'velocity': 0.1,
                    'situational': 0.1,
                }
        
        # Get comparison data
        comparison = self.compare_to(other_stats)
        
        # Create a default response with available set to True for both player types
        default_response = {
            'mss_available': True,
            'mss_score': 50.0,  # Neutral score
            'components': {},
            'weights': weight_config
        }
        
        # Initialize default components for both player types to avoid missing keys
        if self.player_type == 'batter':
            default_components = {
                'batting_avg': 0,
                'on_base_pct': 0,
                'slug_pct': 0,
                'home_runs_rate': 0,
                'strikeout_rate': 0,
                'barrel_rate': 0,
                'launch_speed': 0,
                'situational': 0
            }
        else:  # pitcher
            default_components = {
                'era': 0,
                'whip': 0,
                'k_per_9': 0,
                'bb_per_9': 0,
                'hr_per_9': 0,
                'barrel_rate': 0,
                'velocity': 0,
                'situational': 0
            }
        
        # If comparison isn't available, return the default response with default components
        if not comparison.get('comparison_available', False):
            default_response['components'] = default_components
            return 50.0, default_response
        
        # Initialize score components with defaults to ensure all keys exist
        components = default_components.copy()
        
        # Calculate components based on player type
        if self.player_type == 'batter':
            # Normalize changes to a -1 to 1 scale (where 1 is a positive change)
            
            # Batting average change (normalized to -1 to 1 scale)
            # A 0.050 change in batting average is significant
            ba_change = comparison.get('batting_avg_change', 0)
            components['batting_avg'] = min(max(ba_change / 0.050, -1), 1)
            
            # OBP change
            obp_change = comparison.get('on_base_pct_change', 0)
            components['on_base_pct'] = min(max(obp_change / 0.050, -1), 1)
            
            # SLG change
            slg_change = comparison.get('slug_pct_change', 0)
            components['slug_pct'] = min(max(slg_change / 0.100, -1), 1)
            
            # Home run rate change
            hr_before = self.summary_stats.get('home_runs', 0) / max(self.summary_stats.get('total_abs', 1), 1)
            hr_after = other_stats.summary_stats.get('home_runs', 0) / max(other_stats.summary_stats.get('total_abs', 1), 1)
            hr_change = hr_after - hr_before
            components['home_runs_rate'] = min(max(hr_change / 0.030, -1), 1)
            
            # Strikeout rate change (negative is better)
            k_before = self.summary_stats.get('strikeouts', 0) / max(self.summary_stats.get('total_pas', 1), 1)
            k_after = other_stats.summary_stats.get('strikeouts', 0) / max(other_stats.summary_stats.get('total_pas', 1), 1)
            k_change = k_before - k_after  # Reverse sign since decreasing K rate is good
            components['strikeout_rate'] = min(max(k_change / 0.050, -1), 1)
            
            # Barrel rate change
            barrel_before = self.batted_ball_profile.get('barrel_rate', 0) if self.batted_ball_profile.get('available', False) else 0
            barrel_after = other_stats.batted_ball_profile.get('barrel_rate', 0) if other_stats.batted_ball_profile.get('available', False) else 0
            barrel_change = barrel_after - barrel_before
            components['barrel_rate'] = min(max(barrel_change / 0.030, -1), 1)
            
            # Launch speed change
            speed_before = self.summary_stats.get('avg_launch_speed', 0)
            speed_after = other_stats.summary_stats.get('avg_launch_speed', 0)
            speed_change = speed_after - speed_before
            components['launch_speed'] = min(max(speed_change / 2.0, -1), 1)
            
            # Situational hitting improvement (RISP)
            risp_before = self.situational_stats.get('risp', {}).get('batting_avg', 0)
            risp_after = other_stats.situational_stats.get('risp', {}).get('batting_avg', 0)
            risp_change = risp_after - risp_before
            components['situational'] = min(max(risp_change / 0.070, -1), 1)
            
        else:  # pitcher            
            # ERA change (negative is better)
            era_before = self.summary_stats.get('era', 0)
            era_after = other_stats.summary_stats.get('era', 0)
            era_change = era_before - era_after  # Reverse sign since decreasing ERA is good
            components['era'] = min(max(era_change / 1.0, -1), 1)
            
            # WHIP change (negative is better)
            whip_before = self.summary_stats.get('whip', 0)
            whip_after = other_stats.summary_stats.get('whip', 0)
            whip_change = whip_before - whip_after  # Reverse sign since decreasing WHIP is good
            components['whip'] = min(max(whip_change / 0.300, -1), 1)
            
            # K/9 change
            k9_change = comparison.get('k_per_9_change', 0)
            components['k_per_9'] = min(max(k9_change / 1.5, -1), 1)
            
            # BB/9 change (negative is better)
            bb9_change = -comparison.get('bb_per_9_change', 0)  # Reverse sign since decreasing BB/9 is good
            components['bb_per_9'] = min(max(bb9_change / 1.0, -1), 1)
            
            # HR/9 change (negative is better)
            hr9_change = -comparison.get('hr_per_9_change', 0)  # Reverse sign since decreasing HR/9 is good
            components['hr_per_9'] = min(max(hr9_change / 0.5, -1), 1)
            
            # Velocity change
            velo_before = self.summary_stats.get('avg_velocity', 0)
            velo_after = other_stats.summary_stats.get('avg_velocity', 0)
            velo_change = velo_after - velo_before
            components['velocity'] = min(max(velo_change / 1.0, -1), 1)
            
            # Barrel rate allowed change (negative is better)
            barrel_before = self.batted_ball_profile.get('barrel_rate_allowed', 0) if self.batted_ball_profile.get('available', False) else 0
            barrel_after = other_stats.batted_ball_profile.get('barrel_rate_allowed', 0) if other_stats.batted_ball_profile.get('available', False) else 0
            barrel_change = barrel_before - barrel_after  # Reverse sign since decreasing barrel rate is good
            components['barrel_rate'] = min(max(barrel_change / 0.030, -1), 1)
            
            # Situational pitching improvement (RISP)
            risp_before = self.situational_stats.get('risp', {}).get('batting_avg_against', 0.3)  # Default to 0.3 if not available
            risp_after = other_stats.situational_stats.get('risp', {}).get('batting_avg_against', 0.3)
            risp_change = risp_before - risp_after  # Reverse sign since decreasing is good for pitchers
            components['situational'] = min(max(risp_change / 0.050, -1), 1)
        
        # Calculate weighted average for MSS
        total_weight = sum(weight_config.values())
        weighted_sum = 0
        
        for component, value in components.items():
            if component in weight_config:
                weighted_sum += value * weight_config[component]
        
        # Normalize to a 0-100 scale where 50 is neutral, >50 is positive, <50 is negative
        mss = 50 + (weighted_sum / total_weight) * 50
        
        # Return the MSS score and components
        return mss, {
            'mss_available': True,
            'mss_score': mss,
            'components': components,
            'weights': weight_config
        }


# Helper functions to be used with PlayerStats
def get_player_stats(moment_row, player_col='batter_name', player_type=None, mlbIDs=None):
    """
    Get player statistics for the period before a pivotal moment.
    
    Parameters:
    -----------
    moment_row : pandas.Series or pandas.DataFrame
        Row containing the pivotal moment data
    player_col : str
        Column name containing the player name ('batter_name' or 'pitcher_name')
    player_type : str or None
        Type of player ('batter' or 'pitcher'). If None, inferred from player_col
    mlbIDs : pandas.DataFrame
        DataFrame containing player IDs
        
    Returns:
    --------
    PlayerStats
        PlayerStats object containing the player's statistics
    """
    import pybaseball as pyb
    print("Gathering Player Data")
    
    # If player_type is not provided, infer it from player_col
    if player_type is None:
        player_type = 'batter' if player_col == 'batter_name' else 'pitcher'
    
    moment_row = moment_row.iloc[0] if isinstance(moment_row, pd.DataFrame) else moment_row
    player_name = moment_row[player_col]
    
    try:
        player_id = mlbIDs.loc[mlbIDs['PLAYERNAME'] == player_name, 'MLBID'].values[0]
        
        # Determine the start date (5 seasons prior)
        moment_year = int(moment_row['game_year']) if 'game_year' in moment_row else int(moment_row['game_date'].split('-')[0])
        start_date = f"{moment_year - 5}-01-01"
        
        # Get the statistics
        if player_type == 'batter':
            stats = pyb.statcast_batter(start_date, moment_row['game_date'], player_id=player_id)
        else:  # pitcher
            stats = pyb.statcast_pitcher(start_date, moment_row['game_date'], player_id=player_id)
        
        return PlayerStats(player_name, player_id, stats, start_date, moment_row['game_date'], player_type)
    except Exception as e:
        print(f"Error getting stats for {player_name}: {str(e)}")
        # Return an empty PlayerStats object with the correct player type
        return PlayerStats(player_name, None, pd.DataFrame(), None, None, player_type)


def get_post_moment_stats(moment_row, player_col='batter_name', player_type=None, days_after=30, mlbIDs=None):
    """
    Get player statistics for the period after a pivotal moment.
    
    Parameters:
    -----------
    moment_row : pandas.Series or pandas.DataFrame
        Row containing the pivotal moment data
    player_col : str
        Column name containing the player name ('batter_name' or 'pitcher_name')
    player_type : str or None
        Type of player ('batter' or 'pitcher'). If None, inferred from player_col
    days_after : int
        Number of days after the moment to analyze
    mlbIDs : pandas.DataFrame
        DataFrame containing player IDs
        
    Returns:
    --------
    PlayerStats
        PlayerStats object containing the player's statistics
    """
    import pybaseball as pyb
    from datetime import datetime, timedelta
    print("Gathering Player Data")
    
    # If player_type is not provided, infer it from player_col
    if player_type is None:
        player_type = 'batter' if player_col == 'batter_name' else 'pitcher'
    
    moment_row = moment_row.iloc[0] if isinstance(moment_row, pd.DataFrame) else moment_row
    player_name = moment_row[player_col]
    
    try:
        player_id = mlbIDs.loc[mlbIDs['PLAYERNAME'] == player_name, 'MLBID'].values[0]
        
        # Calculate end date (days_after days after the moment)
        start_date = moment_row['game_date']
        date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = (date_obj + timedelta(days=days_after)).strftime('%Y-%m-%d')
        
        # Get the statistics
        if player_type == 'batter':
            stats = pyb.statcast_batter(start_date, end_date, player_id=player_id)
        else:  # pitcher
            stats = pyb.statcast_pitcher(start_date, end_date, player_id=player_id)
        
        return PlayerStats(player_name, player_id, stats, start_date, end_date, player_type)
    except Exception as e:
        print(f"Error getting post-moment stats for {player_name}: {str(e)}")
        # Return an empty PlayerStats object with the correct player type
        return PlayerStats(player_name, None, pd.DataFrame(), None, None, player_type)


def calculate_momentum_shift_score(moment_row, days_before=None, days_after=30, mlbIDs=None):
    """
    Calculate the Momentum Shift Score for a pivotal moment.
    
    Parameters:
    -----------
    moment_row : pandas.Series or pandas.DataFrame
        Row containing the pivotal moment data
    days_before : int or None
        Number of days before the moment to analyze (if None, uses 5 seasons)
    days_after : int
        Number of days after the moment to analyze
    mlbIDs : pandas.DataFrame
        DataFrame containing player IDs
        
    Returns:
    --------
    dict
        Dictionary containing MSS scores for both batter and pitcher
    """
    moment_row = moment_row.iloc[0] if isinstance(moment_row, pd.DataFrame) else moment_row
    
    results = {
        'moment_date': moment_row['game_date'],
        'moment_type': moment_row['events'],
        'wpa_change': moment_row['delta_home_win_exp_abs'] if 'delta_home_win_exp_abs' in moment_row else None,
        'batter': {
            'name': moment_row['batter_name'],
            'mss': None,
            'details': None
        },
        'pitcher': {
            'name': moment_row['pitcher_name'],
            'mss': None,
            'details': None
        }
    }
    
    # Calculate MSS for batter
    batter_pre = get_player_stats(moment_row, player_col='batter_name', player_type='batter', mlbIDs=mlbIDs)
    batter_post = get_post_moment_stats(moment_row, player_col='batter_name', player_type='batter', 
                                       days_after=days_after, mlbIDs=mlbIDs)
    
    if batter_pre.data_available and batter_post.data_available:
        batter_mss, batter_details = batter_pre.calculate_mss(batter_post)
        results['batter']['mss'] = batter_mss
        results['batter']['details'] = batter_details
    else:
        # Provide default details even if data isn't available
        _, default_details = PlayerStats('', None, pd.DataFrame(), None, None, 'batter').calculate_mss(
            PlayerStats('', None, pd.DataFrame(), None, None, 'batter')
        )
        results['batter']['mss'] = 50.0
        results['batter']['details'] = default_details
    
    # Calculate MSS for pitcher
    pitcher_pre = get_player_stats(moment_row, player_col='pitcher_name', player_type='pitcher', mlbIDs=mlbIDs)
    pitcher_post = get_post_moment_stats(moment_row, player_col='pitcher_name', player_type='pitcher', 
                                        days_after=days_after, mlbIDs=mlbIDs)
    
    if pitcher_pre.data_available and pitcher_post.data_available:
        pitcher_mss, pitcher_details = pitcher_pre.calculate_mss(pitcher_post)
        results['pitcher']['mss'] = pitcher_mss
        results['pitcher']['details'] = pitcher_details
    else:
        # Provide default details even if data isn't available
        _, default_details = PlayerStats('', None, pd.DataFrame(), None, None, 'pitcher').calculate_mss(
            PlayerStats('', None, pd.DataFrame(), None, None, 'pitcher')
        )
        results['pitcher']['mss'] = 50.0
        results['pitcher']['details'] = default_details
    
    return results