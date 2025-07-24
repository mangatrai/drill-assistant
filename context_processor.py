"""
Context Processor for Oil & Gas Data
Extracts meaningful insights and creates structured documents for vector storage
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import re

@dataclass
class ContextualDocument:
    """Structured document for vector storage"""
    content: str
    metadata: Dict[str, Any]
    document_type: str
    source: str
    timestamp: str

class OilGasContextProcessor:
    """Processes parsed oil & gas data into contextual documents"""
    
    def __init__(self, parsed_data: Dict[str, Any]):
        self.parsed_data = parsed_data
        self.documents = []
        
    def process_all_data(self) -> List[ContextualDocument]:
        """Process all data into contextual documents"""
        print("Processing data into contextual documents...")
        
        # Process different data types
        self._process_well_summaries()
        self._process_formation_analysis()
        self._process_petrophysical_insights()
        self._process_seismic_metadata()
        self._process_well_trajectories()
        self._process_field_overview()
        
        print(f"Generated {len(self.documents)} contextual documents")
        return self.documents
    
    def _process_well_summaries(self):
        """Create comprehensive well summaries"""
        print("Creating well summaries...")
        
        for well_name, well_data in self.parsed_data['wells'].items():
            # Basic well information
            content = f"""
            Well Summary: {well_name}
            
            Field: {well_data.field}
            Company: {well_data.company}
            Wellbore: {well_data.wellbore_name}
            
            Location Information:
            - Surface Coordinates: E/W {well_data.coordinates.get('surface_ew', 'N/A')}m, N/S {well_data.coordinates.get('surface_ns', 'N/A')}m
            - Latitude: {well_data.coordinates.get('latitude', 'N/A')}
            - Longitude: {well_data.coordinates.get('longitude', 'N/A')}
            
            Data Availability:
            - Survey Data: {'Available' if well_data.survey_data is not None else 'Not Available'}
            - Formation Picks: {'Available' if well_data.formation_picks is not None else 'Not Available'}
            - Petrophysical Data: {'Available' if well_data.petrophysical_data is not None else 'Not Available'}
            """
            
            # Add survey statistics if available
            if well_data.survey_data is not None and not well_data.survey_data.empty:
                survey_stats = self._calculate_survey_statistics(well_data.survey_data)
                content += f"""
                
            Survey Statistics:
            - Total Depth: {survey_stats['max_depth']:.1f}m
            - Maximum Inclination: {survey_stats['max_inclination']:.1f}°
            - Maximum Azimuth: {survey_stats['max_azimuth']:.1f}°
            - Survey Points: {len(well_data.survey_data)}
            """
            
            # Add formation information if available
            if well_data.formation_picks is not None and not well_data.formation_picks.empty:
                formations = self._extract_formation_info(well_data.formation_picks)
                content += f"""
                
            Geological Formations:
            {formations}
            """
            
            document = ContextualDocument(
                content=content.strip(),
                metadata={
                    'well_name': well_name,
                    'field': well_data.field,
                    'company': well_data.company,
                    'document_type': 'well_summary',
                    'has_survey': well_data.survey_data is not None,
                    'has_picks': well_data.formation_picks is not None,
                    'has_petrophysical': well_data.petrophysical_data is not None
                },
                document_type='well_summary',
                source=f'well_{well_name}',
                timestamp=datetime.now().isoformat()
            )
            
            self.documents.append(document)
    
    def _process_formation_analysis(self):
        """Create formation analysis documents"""
        print("Creating formation analysis...")
        
        # Collect all formation data across wells
        all_formations = {}
        
        for well_name, well_data in self.parsed_data['wells'].items():
            if well_data.formation_picks is not None:
                for _, pick in well_data.formation_picks.iterrows():
                    formation_name = pick['Surface_name']
                    if formation_name not in all_formations:
                        all_formations[formation_name] = []
                    
                    all_formations[formation_name].append({
                        'well': well_name,
                        'depth_md': pick['MD'],
                        'depth_tvd': pick['TVD'],
                        'depth_tvdss': pick['TVDSS'],
                        'twt': pick['TWT'],
                        'dip': pick['Dip'],
                        'azimuth': pick['Azi']
                    })
        
        # Create formation analysis documents
        for formation_name, picks in all_formations.items():
            if len(picks) > 0:
                # Calculate formation statistics
                depths_md = [p['depth_md'] for p in picks if pd.notna(p['depth_md'])]
                depths_tvd = [p['depth_tvd'] for p in picks if pd.notna(p['depth_tvd'])]
                twts = [p['twt'] for p in picks if pd.notna(p['twt'])]
                
                content = f"""
                Formation Analysis: {formation_name}
                
                Occurrence: Found in {len(picks)} wells
                Wells: {', '.join(set(p['well'] for p in picks))}
                
                Depth Statistics (Measured Depth):
                - Average: {np.mean(depths_md):.1f}m
                - Range: {min(depths_md):.1f}m - {max(depths_md):.1f}m
                - Standard Deviation: {np.std(depths_md):.1f}m
                
                Depth Statistics (True Vertical Depth):
                - Average: {np.mean(depths_tvd):.1f}m
                - Range: {min(depths_tvd):.1f}m - {max(depths_tvd):.1f}m
                - Standard Deviation: {np.std(depths_tvd):.1f}m
                """
                
                if twts:
                    content += f"""
                Time Statistics (Two-Way Time):
                - Average: {np.mean(twts):.1f}ms
                - Range: {min(twts):.1f}ms - {max(twts):.1f}ms
                - Standard Deviation: {np.std(twts):.1f}ms
                """
                
                # Add geological context
                geological_context = self._get_geological_context(formation_name)
                if geological_context:
                    content += f"\nGeological Context:\n{geological_context}"
                
                document = ContextualDocument(
                    content=content.strip(),
                    metadata={
                        'formation_name': formation_name,
                        'well_count': len(set(p['well'] for p in picks)),
                        'document_type': 'formation_analysis',
                        'avg_depth_md': np.mean(depths_md),
                        'avg_depth_tvd': np.mean(depths_tvd)
                    },
                    document_type='formation_analysis',
                    source=f'formation_{formation_name}',
                    timestamp=datetime.now().isoformat()
                )
                
                self.documents.append(document)
    
    def _process_petrophysical_insights(self):
        """Create petrophysical analysis documents"""
        print("Creating petrophysical insights...")
        
        for well_name, well_data in self.parsed_data['wells'].items():
            if well_data.petrophysical_data:
                petro_data = well_data.petrophysical_data
                
                content = f"""
                Petrophysical Analysis: {well_name}
                
                Available Data Types:
                """
                
                # Input curves
                if petro_data.get('input_curves'):
                    content += "\nInput Curves:\n"
                    for mnem, curve_info in petro_data['input_curves'].items():
                        content += f"- {mnem}: {curve_info.get('description', 'No description')} ({curve_info.get('unit', 'No unit')})\n"
                
                # Output curves
                if petro_data.get('output_curves'):
                    content += "\nOutput Curves:\n"
                    for mnem, curve_info in petro_data['output_curves'].items():
                        content += f"- {mnem}: {curve_info.get('description', 'No description')} ({curve_info.get('unit', 'No unit')})\n"
                
                # LAS data
                if petro_data.get('las_data'):
                    las_data = petro_data['las_data']
                    content += f"""
                
                LAS Data Information:
                - Depth Range: {las_data.get('depth_range', {}).get('start', 'N/A')}m - {las_data.get('depth_range', {}).get('stop', 'N/A')}m
                - Step Size: {las_data.get('depth_range', {}).get('step', 'N/A')}m
                - Available Curves: {len(las_data.get('curve_info', {}))}
                """
                    
                    # Add curve information
                    if las_data.get('curve_info'):
                        content += "\nCurve Details:\n"
                        for mnem, curve_info in las_data['curve_info'].items():
                            content += f"- {mnem}: {curve_info.get('description', 'No description')} ({curve_info.get('unit', 'No unit')})\n"
                
                # Add interpretation insights
                interpretation_insights = self._generate_petrophysical_insights(petro_data)
                if interpretation_insights:
                    content += f"\nInterpretation Insights:\n{interpretation_insights}"
                
                document = ContextualDocument(
                    content=content.strip(),
                    metadata={
                        'well_name': well_name,
                        'document_type': 'petrophysical_analysis',
                        'input_curves_count': len(petro_data.get('input_curves', {})),
                        'output_curves_count': len(petro_data.get('output_curves', {})),
                        'has_las_data': 'las_data' in petro_data
                    },
                    document_type='petrophysical_analysis',
                    source=f'petrophysical_{well_name}',
                    timestamp=datetime.now().isoformat()
                )
                
                self.documents.append(document)
    
    def _process_seismic_metadata(self):
        """Create seismic data analysis documents"""
        print("Creating seismic analysis...")
        
        for filename, seismic_info in self.parsed_data['seismic'].items():
            content = f"""
            Seismic Data Analysis: {filename}
            
            File Information:
            - Size: {seismic_info['size_mb']:.1f} MB
            - Survey Type: {seismic_info['survey_type']}
            - Migration: {seismic_info['migration']}
            - Algorithm: {seismic_info['algorithm']}
            
            Technical Details:
            - This is a 3D post-stack migrated seismic volume
            - Uses Kirchhoff pre-stack depth migration (PSDM)
            - Suitable for structural interpretation and reservoir characterization
            - Large file size indicates high-resolution data
            """
            
            # Add seismic interpretation guidance
            interpretation_guide = self._generate_seismic_interpretation_guide(seismic_info)
            content += f"\nInterpretation Guidance:\n{interpretation_guide}"
            
            document = ContextualDocument(
                content=content.strip(),
                metadata={
                    'filename': filename,
                    'document_type': 'seismic_analysis',
                    'size_mb': seismic_info['size_mb'],
                    'survey_type': seismic_info['survey_type'],
                    'migration': seismic_info['migration']
                },
                document_type='seismic_analysis',
                source=f'seismic_{filename}',
                timestamp=datetime.now().isoformat()
            )
            
            self.documents.append(document)
    
    def _process_well_trajectories(self):
        """Create well trajectory analysis documents"""
        print("Creating trajectory analysis...")
        
        for well_name, well_data in self.parsed_data['wells'].items():
            if well_data.survey_data is not None and not well_data.survey_data.empty:
                trajectory_analysis = self._analyze_well_trajectory(well_data.survey_data)
                
                content = f"""
                Well Trajectory Analysis: {well_name}
                
                Trajectory Characteristics:
                - Total Measured Depth: {trajectory_analysis['total_md']:.1f}m
                - Total Vertical Depth: {trajectory_analysis['total_tvd']:.1f}m
                - Maximum Inclination: {trajectory_analysis['max_inclination']:.1f}°
                - Maximum Azimuth: {trajectory_analysis['max_azimuth']:.1f}°
                - Maximum Dogleg Severity: {trajectory_analysis['max_dls']:.2f}°/30m
                
                Trajectory Type: {trajectory_analysis['trajectory_type']}
                
                Key Features:
                {trajectory_analysis['key_features']}
                
                Drilling Implications:
                {trajectory_analysis['drilling_implications']}
                """
                
                document = ContextualDocument(
                    content=content.strip(),
                    metadata={
                        'well_name': well_name,
                        'document_type': 'trajectory_analysis',
                        'total_md': trajectory_analysis['total_md'],
                        'total_tvd': trajectory_analysis['total_tvd'],
                        'max_inclination': trajectory_analysis['max_inclination'],
                        'trajectory_type': trajectory_analysis['trajectory_type']
                    },
                    document_type='trajectory_analysis',
                    source=f'trajectory_{well_name}',
                    timestamp=datetime.now().isoformat()
                )
                
                self.documents.append(document)
    
    def _process_field_overview(self):
        """Create comprehensive field overview document"""
        print("Creating field overview...")
        
        summary = self.parsed_data['summary']
        
        content = f"""
        Volve Field Overview
        
        Field Statistics:
        - Total Wells: {summary['total_wells']}
        - Wells with Survey Data: {summary['wells_with_survey']}
        - Wells with Formation Picks: {summary['wells_with_picks']}
        - Wells with Petrophysical Data: {summary['wells_with_petrophysical']}
        - Seismic Files: {summary['total_seismic_files']}
        - Total Seismic Data Size: {summary['total_seismic_size_gb']:.1f} GB
        
        Field Characteristics:
        - Location: North Sea, Norwegian Continental Shelf
        - Operator: Statoil (now Equinor)
        - Field Type: Oil field
        - Reservoir: Jurassic Hugin Formation
        
        Data Quality Assessment:
        - Comprehensive well coverage with multiple data types
        - High-quality 3D seismic data available
        - Detailed petrophysical interpretations
        - Complete formation picks for stratigraphic analysis
        
        Exploration and Development Context:
        - Volve was a mature oil field with extensive development history
        - Data represents comprehensive field characterization
        - Suitable for reservoir modeling and production optimization studies
        - Excellent dataset for machine learning applications in oil & gas
        """
        
        document = ContextualDocument(
            content=content.strip(),
            metadata={
                'document_type': 'field_overview',
                'total_wells': summary['total_wells'],
                'total_seismic_files': summary['total_seismic_files'],
                'total_seismic_size_gb': summary['total_seismic_size_gb']
            },
            document_type='field_overview',
            source='field_overview',
            timestamp=datetime.now().isoformat()
        )
        
        self.documents.append(document)
    
    def _calculate_survey_statistics(self, survey_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate survey statistics"""
        return {
            'max_depth': survey_data['MD'].max(),
            'max_inclination': survey_data['Inc'].max(),
            'max_azimuth': survey_data['Azim'].max(),
            'avg_inclination': survey_data['Inc'].mean(),
            'avg_azimuth': survey_data['Azim'].mean()
        }
    
    def _extract_formation_info(self, picks_data: pd.DataFrame) -> str:
        """Extract formation information from picks"""
        formations = picks_data['Surface_name'].unique()
        formation_info = []
        
        for formation in formations:
            formation_picks = picks_data[picks_data['Surface_name'] == formation]
            avg_depth = formation_picks['MD'].mean()
            formation_info.append(f"- {formation}: Average depth {avg_depth:.1f}m")
        
        return '\n'.join(formation_info)
    
    def _get_geological_context(self, formation_name: str) -> str:
        """Get geological context for formation"""
        geological_contexts = {
            'Seabed': 'Modern seafloor surface',
            'NORDLAND GP.': 'Quaternary to Pliocene sediments',
            'Utsira Fm.': 'Pliocene deep marine sands',
            'HORDALAND GP.': 'Miocene to Oligocene sediments',
            'Ty Fm.': 'Paleocene deep marine sands',
            'SHETLAND GP.': 'Cretaceous to Paleocene sediments',
            'Ekofisk Fm.': 'Cretaceous chalk formation',
            'Hod Fm.': 'Cretaceous chalk and marl',
            'Draupne Fm.': 'Upper Jurassic organic-rich shale',
            'Heather Fm.': 'Middle Jurassic marine shale',
            'Hugin Fm.': 'Middle Jurassic reservoir sandstone',
            'Sleipner Fm.': 'Middle Jurassic sandstone',
            'Skagerrak Fm.': 'Triassic continental deposits',
            'Smith Bank Fm.': 'Triassic continental deposits'
        }
        
        for key, context in geological_contexts.items():
            if key in formation_name:
                return context
        
        return "Formation context not available"
    
    def _generate_petrophysical_insights(self, petro_data: Dict) -> str:
        """Generate petrophysical interpretation insights"""
        insights = []
        
        # Analyze available curves
        input_curves = petro_data.get('input_curves', {})
        output_curves = petro_data.get('output_curves', {})
        
        if 'GR' in input_curves:
            insights.append("- Gamma Ray (GR) available for lithology identification")
        if 'DT' in input_curves:
            insights.append("- Sonic (DT) available for porosity estimation")
        if 'NPHI' in input_curves:
            insights.append("- Neutron porosity available for porosity analysis")
        if 'RHOB' in input_curves:
            insights.append("- Density (RHOB) available for density-porosity analysis")
        if 'RT' in input_curves:
            insights.append("- Resistivity (RT) available for water saturation analysis")
        
        if 'PHIF' in output_curves:
            insights.append("- Effective porosity (PHIF) calculated")
        if 'SW' in output_curves:
            insights.append("- Water saturation (SW) calculated")
        if 'VSH' in output_curves:
            insights.append("- Shale volume (VSH) calculated")
        if 'KLOGH' in output_curves:
            insights.append("- Permeability (KLOGH) estimated")
        
        return '\n'.join(insights) if insights else "Standard petrophysical interpretation available"
    
    def _generate_seismic_interpretation_guide(self, seismic_info: Dict) -> str:
        """Generate seismic interpretation guidance"""
        return """
        - Use for structural interpretation of faults and folds
        - Identify reservoir boundaries and fluid contacts
        - Map stratigraphic features and depositional environments
        - Perform attribute analysis for reservoir characterization
        - Integrate with well data for calibration and validation
        """
    
    def _analyze_well_trajectory(self, survey_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze well trajectory characteristics"""
        max_inclination = survey_data['Inc'].max()
        max_azimuth = survey_data['Azim'].max()
        max_dls = survey_data['DLS'].max()
        total_md = survey_data['MD'].max()
        total_tvd = survey_data['TVD'].max()
        
        # Determine trajectory type
        if max_inclination < 5:
            trajectory_type = "Vertical Well"
        elif max_inclination < 30:
            trajectory_type = "Slightly Deviated Well"
        elif max_inclination < 60:
            trajectory_type = "Deviated Well"
        else:
            trajectory_type = "Horizontal Well"
        
        # Key features
        key_features = []
        if max_inclination > 80:
            key_features.append("- High-angle or horizontal section present")
        if max_dls > 3:
            key_features.append("- High dogleg severity sections")
        if total_md - total_tvd > 1000:
            key_features.append("- Significant lateral displacement")
        
        # Drilling implications
        implications = []
        if max_inclination > 60:
            implications.append("- Requires specialized drilling techniques")
        if max_dls > 3:
            implications.append("- High stress on drill string and casing")
        if total_md - total_tvd > 1000:
            implications.append("- Extended reach drilling techniques used")
        
        return {
            'total_md': total_md,
            'total_tvd': total_tvd,
            'max_inclination': max_inclination,
            'max_azimuth': max_azimuth,
            'max_dls': max_dls,
            'trajectory_type': trajectory_type,
            'key_features': '\n'.join(key_features) if key_features else "- Standard well trajectory",
            'drilling_implications': '\n'.join(implications) if implications else "- Standard drilling practices"
        }

if __name__ == "__main__":
    # Load parsed data
    with open("parsed_data.json", "r") as f:
        parsed_data = json.load(f)
    
    # Process into contextual documents
    processor = OilGasContextProcessor(parsed_data)
    documents = processor.process_all_data()
    
    # Save contextual documents
    with open("contextual_documents.json", "w") as f:
        json.dump([{
            'content': doc.content,
            'metadata': doc.metadata,
            'document_type': doc.document_type,
            'source': doc.source,
            'timestamp': doc.timestamp
        } for doc in documents], f, indent=2)
    
    print(f"Generated {len(documents)} contextual documents")
    print("Context processing completed!") 