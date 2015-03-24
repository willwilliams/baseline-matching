require 'paths'
require 'parallel'

paths.dofile('parallelmain.lua')
ok, err = pcall(doMain())
if not ok then 
	print("Error!")
	print(err) 
	parallel.close() 
end
