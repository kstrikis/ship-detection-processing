"""
Module for reading SAR data files using sarpy, focused on CPHD and SICD formats.
"""

import os
import logging
from typing import Union, Dict, Tuple, Optional, List, Any

import numpy as np
from sarpy.io.general.utils import get_filename_extension
from sarpy.io.phase_history.cphd import CPHDReader
from sarpy.io.complex.sicd import SICDReader
from sarpy.io.phase_history.converter import open_phase_history

# Setup logging
logger = logging.getLogger(__name__)

class SARDataReader:
    """
    Class for reading SAR data from various file formats, with a focus on CPHD and SICD.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the SAR data reader.
        
        Parameters
        ----------
        file_path : str
            Path to the SAR data file.
        """
        self.file_path = file_path
        self.extension = get_filename_extension(file_path)
        self.reader = None
        self.metadata = None
        self._load_data()
    
    def _load_data(self) -> None:
        """
        Load the SAR data file based on its extension.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found.")
        
        try:
            # Determine file type and open with appropriate reader
            if self.extension.lower() == '.cphd':
                logger.info(f"Opening CPHD file: {self.file_path}")
                # Use the phase history converter to open the file
                self.reader = open_phase_history(self.file_path)
                if isinstance(self.reader, CPHDReader):
                    self.metadata = self.reader.cphd_meta
                else:
                    logger.warning("Reader is not a CPHDReader as expected")
            elif self.extension.lower() == '.nitf':
                logger.info(f"Opening NITF (SICD) file: {self.file_path}")
                # Try opening as SICD
                from sarpy.io.complex.converter import open_complex
                self.reader = open_complex(self.file_path)
                if hasattr(self.reader, 'sicd_meta'):
                    self.metadata = self.reader.sicd_meta
            else:
                logger.info(f"Attempting to open file: {self.file_path}")
                # Try a general open approach
                from sarpy.io.general.converter import open_general
                self.reader = open_general(self.file_path)
                if hasattr(self.reader, 'sicd_meta'):
                    self.metadata = self.reader.sicd_meta
                elif hasattr(self.reader, 'cphd_meta'):
                    self.metadata = self.reader.cphd_meta
                else:
                    logger.warning("Could not extract metadata from file")
        except Exception as e:
            logger.error(f"Failed to open file {self.file_path}: {str(e)}")
            raise
    
    def get_metadata(self) -> Dict:
        """
        Get the metadata from the SAR data file.
        
        Returns
        -------
        Dict
            The metadata dictionary.
        """
        return self.metadata
    
    def read_cphd_signal_data(self, channel_id: Union[int, str] = 0) -> np.ndarray:
        """
        Read signal data from a CPHD file.
        
        Parameters
        ----------
        channel_id : Union[int, str], optional
            The channel ID or index to read, by default 0
            
        Returns
        -------
        np.ndarray
            The signal data as a complex array.
        """
        if not isinstance(self.reader, CPHDReader):
            raise TypeError("Reader is not a CPHDReader")
        
        # Read the signal data for the specified channel
        return self.reader.read_signal_block()[channel_id]
    
    def read_pvp_data(self, channel_id: Union[int, str] = 0) -> Dict[str, np.ndarray]:
        """
        Read Per Vector Parameter (PVP) data from a CPHD file.
        
        Parameters
        ----------
        channel_id : Union[int, str], optional
            The channel ID or index to read, by default 0
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of PVP arrays.
        """
        if not isinstance(self.reader, CPHDReader):
            raise TypeError("Reader is not a CPHDReader")
        
        # Read the PVP array for the specified channel
        pvp_array = self.reader.read_pvp_array(channel_id)
        
        # Extract individual PVP variables
        result = {}
        pvp_variables = ['TxTime', 'TxPos', 'RcvTime', 'RcvPos', 'SRPPos', 'SCSS']
        
        for variable in pvp_variables:
            try:
                result[variable] = self.reader.read_pvp_variable(variable, channel_id)
            except Exception as e:
                logger.warning(f"Failed to read PVP variable {variable}: {str(e)}")
        
        return result
    
    def read_sicd_data(self, index: int = 0) -> np.ndarray:
        """
        Read image data from a SICD file.
        
        Parameters
        ----------
        index : int, optional
            The image index to read, by default 0
            
        Returns
        -------
        np.ndarray
            The complex image data.
        """
        if not hasattr(self.reader, 'sicd_meta'):
            raise TypeError("Reader does not have SICD metadata")
        
        # Read the full image data
        return self.reader.read_chip()[0]
    
    def close(self) -> None:
        """
        Close the reader.
        """
        if self.reader is not None:
            self.reader.close() 