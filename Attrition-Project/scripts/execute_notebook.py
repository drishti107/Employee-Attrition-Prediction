"""Execute the walkthrough in-process, capturing actual outputs without sockets.
The notebook remains runnable using a normal Jupyter kernel as well.
"""
from pathlib import Path
import ast,os,sys,json
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/Attrition_Project_Walkthrough.ipynb'
nb=nbformat.read(path,as_version=4)
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
shell=InteractiveShell.instance();ns=shell.user_ns;ns['display']=display
count=0
for cell in nb.cells:
    if cell.cell_type!='code':continue
    count+=1;cell.execution_count=count;cell.outputs=[]
    tree=ast.parse(cell.source)
    with capture_output() as captured:
        if tree.body and isinstance(tree.body[-1],ast.Expr):
            prefix=ast.Module(body=tree.body[:-1],type_ignores=[])
            exec(compile(prefix,str(path),'exec'),ns)
            value=eval(compile(ast.Expression(tree.body[-1].value),str(path),'eval'),ns)
            if value is not None:display(value)
        else:exec(compile(tree,str(path),'exec'),ns)
    if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
nb.metadata['execution']={'method':'In-process IPython with actual captured outputs; no kernel socket required','code_cells_executed':count,'status':'PASS'}
nbformat.write(nb,path)
print('Executed',count,'cells successfully:',path.name)
