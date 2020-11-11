#!/usr/bin/env ruby
#coding: UTF-8

require 'json'

json_data = open(ARGV[0]) do |io|
  JSON.load(io)
end

coordinates = json_data['features'][0]['geometry']['coordinates'][0]

puts "none\n1"

coordinates.each do |coord|
  y = sprintf('%e', coord[0])
  x = sprintf('%e', coord[1])
  coord_data = " #{y} #{x}"
  puts coord_data.gsub("e+","E+")
end

puts "END\nEND"