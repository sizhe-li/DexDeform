echo Test1
go foo.sh,"python bar.py"
echo Test2
go run.py,run2.py --opts --name name1,name2 --seed "@range(1,3)" --path {seed}_{name}
echo Test3
go run.py -o --name name --seed @"range(1,3)" --path {seed}_{name}
echo TestZIP1
go run.py --zip -o --name name1,name2 --seed @"range(1,3)" --path "{seed}_{name}"
echo TestZIP2
go run.py,run2.py --zip -o --name name1,name2 --seed @"range(1,3)" --path "{seed}_{name}"
echo TestFree
go run.py,run2.py --xxx y -o name {xxx} # we don't allow this.. if you can decide it, please enter it twice..
echo TestFree2
go run.py,run2.py --output -o --xxx 1,3 name {xxx} # we don't allow this.. if you can decide it, please enter it twice..
