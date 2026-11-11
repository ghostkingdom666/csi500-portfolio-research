"""Portable task entry point: acquire, rebuild, validate (Python 3.12+)."""
from pathlib import Path
import argparse,subprocess,sys

ROOT=Path(__file__).resolve().parent

def run(script,*args):
    subprocess.run([sys.executable,'-u',str(ROOT/'src'/script),*args],check=True)

def main():
    ap=argparse.ArgumentParser(description='01任务卡数据处理入口')
    ap.add_argument('command',choices=['acquire','rebuild','validate'])
    ap.add_argument('--output',type=Path,default=ROOT/'data')
    args=ap.parse_args()
    if args.command=='acquire':
        if (ROOT/'data/manifest.json').exists():
            ap.error('This snapshot is frozen. Copy the package to a NEW version directory and remove that copy\'s raw/data/config/logs before a fresh acquisition; never overwrite this snapshot.')
        run('prepare.py');run('acquire.py','--stage','sample');run('acquire.py','--stage','all');run('enrich.py');run('verify_announcements.py')
        run('build.py','--output',str(args.output));run('validate.py','--data',str(args.output))
    elif args.command=='rebuild':run('build.py','--output',str(args.output))
    else:run('validate.py','--data',str(args.output))

if __name__=='__main__':main()
