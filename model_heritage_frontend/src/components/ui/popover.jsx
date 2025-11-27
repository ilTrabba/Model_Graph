import React, { useState, useRef, useEffect } from 'react'
import { clsx } from 'clsx'

const Popover = ({ children }) => {
  const [isOpen, setIsOpen] = useState(false)
  
  return (
    <PopoverContext.Provider value={{ isOpen, setIsOpen }}>
      <div className="relative inline-block">
        {children}
      </div>
    </PopoverContext.Provider>
  )
}

const PopoverContext = React.createContext({ isOpen: false, setIsOpen: () => {} })

const PopoverTrigger = React.forwardRef(({ children, asChild, ...props }, ref) => {
  const { isOpen, setIsOpen } = React.useContext(PopoverContext)
  
  const handleClick = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsOpen(!isOpen)
  }
  
  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, {
      ...props,
      ref,
      onClick: handleClick,
      'aria-expanded': isOpen,
    })
  }
  
  return (
    <button ref={ref} onClick={handleClick} {...props}>
      {children}
    </button>
  )
})
PopoverTrigger.displayName = "PopoverTrigger"

const PopoverContent = React.forwardRef(({ children, className, align = "start", ...props }, ref) => {
  const { isOpen, setIsOpen } = React.useContext(PopoverContext)
  const contentRef = useRef(null)
  
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (contentRef.current && !contentRef.current.contains(event.target)) {
        // Check if click is on trigger - close if click is outside the popover root
        const popoverRoot = contentRef.current.closest('.relative.inline-block')
        if (!popoverRoot || !popoverRoot.contains(event.target)) {
          setIsOpen(false)
        }
      }
    }
    
    if (isOpen) {
      // Delay adding listener to prevent immediate close
      setTimeout(() => {
        document.addEventListener('mousedown', handleClickOutside)
      }, 0)
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen, setIsOpen])
  
  if (!isOpen) return null
  
  const alignClasses = {
    start: 'left-0',
    center: 'left-1/2 -translate-x-1/2',
    end: 'right-0'
  }
  
  return (
    <div
      ref={(node) => {
        contentRef.current = node
        if (typeof ref === 'function') ref(node)
        else if (ref) ref.current = node
      }}
      className={clsx(
        "absolute z-50 mt-2 rounded-md border border-gray-200 bg-white shadow-lg",
        alignClasses[align],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
})
PopoverContent.displayName = "PopoverContent"

export { Popover, PopoverTrigger, PopoverContent }
