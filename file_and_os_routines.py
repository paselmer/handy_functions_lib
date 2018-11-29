# Routines to perform typical OS and file tasks, such as creating a list
# of files matching a certain regular expression, or deleting a file.

import pdb
import struct as struct
import csv
import datetime as DT
import xdrlib
import os
from subprocess import check_output
import numpy as np

def delete_file_in_Windows(file_name):
    """ Delete a file in Windows OS. """
    
    cmd = 'del ' + file_name
    cmd_feedback = check_output(cmd, shell=True)
    
    
def delete_file_in_Unix(file_name):
    """ Delete a file in Unix-type OS. """
    
    cmd = 'rm ' + file_name
    cmd_feedback = check_output(cmd, shell=True)    


def delete_file(file_name):
    """ Calls appropriate function depending on OS. """
    if (os.name != 'nt'):                                # Linux/Unix
        delete_file_in_Unix(file_name)
    else:                                                # Windows
        delete_file_in_Windows(file_name)    
    

def create_a_file_list_in_Windows(file_list,raw_dir,search_str):    
    """ This code  will create a file list in the current directory
        with the given input name. The other input text string is 
        so code can be reused to search for different systematically
        named types of files.
    """
    
    cmd = 'type nul > ' + file_list
    cmd_feedback = check_output(cmd, shell=True)
    cmd = 'del ' + file_list
    cmd_feedback = check_output(cmd, shell=True)
    cmd = 'dir ' + raw_dir + search_str + ' /B/S/ON > ' + file_list 
    cmd_feedback = check_output(cmd, shell=True)
    
    
def create_a_file_list_in_Unix(file_list,raw_dir,search_str):
    """ This code will create a file list in the current directory
        with the given input name. The other input text string is
        so code can be reused to search for different systematically
        named types of files.
    """
    
    # *** NOTE *** 
    # This code relies on an external C-shell script.

    cmd = 'touch ' + file_list
    cmd_feedback = check_output(cmd, shell=True)
    cmd = 'rm ' + file_list
    cmd_feedback = check_output(cmd, shell=True)
    cmd = './make_file_list_unix ' + raw_dir + ' ' + search_str + ' > ' + file_list
    cmd_feedback = check_output(cmd, shell=True)   
    