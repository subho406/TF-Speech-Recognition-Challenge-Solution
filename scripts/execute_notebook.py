#!/usr/bin/env python
import os
import glob

if __name__=='__main__':
    print('Searching for notebooks in the notebooks directory')
    if(os.path.isdir('../notebooks')==False):
        print('Notebook Directory not found! Exiting')
        exit(0)
    notebooks=glob.glob('../notebooks/*.ipynb')
    if(len(notebooks)==0):
        print('No Notebooks found! Exiting.')
        exit(0)
    print('Select a notebook to run. Results will be logged to <notebook_name>.log in the results directory\n')
    for i in range(len(notebooks)):
        print('%d. %s'%(i+1,os.path.basename(notebooks[i])))

    try:
        option = int(input('\nEnter option: '))
        if(option>len(notebooks)):
            assert Exception
        print('Executing notebook %s' % os.path.basename(notebooks[option - 1]))
        selected_notebook = notebooks[option - 1].replace(' ', '\ ')
        result_file_name=os.path.splitext(os.path.basename(selected_notebook))[0]
        os.system('jupyter nbconvert --to script --execute --stdout %s | python -u 2>&1 | tee  ../results/%s.log &' % (selected_notebook,result_file_name))
        print('Process started!')
    except Exception as e:
        print('Invalid option! Existing.')
        exit(0)
